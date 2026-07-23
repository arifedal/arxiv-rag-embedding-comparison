"""
PHASE 3: LLM Model Comparison
Süre: ~35 saat
Amaç: RAG ile farklı LLM'leri karşılaştırmak
"""

import pandas as pd
import numpy as np
import json
import time
from typing import List, Dict, Tuple, Optional
import os
from tqdm import tqdm
import faiss
from dataclasses import dataclass
import matplotlib.pyplot as plt

# LLM için llama-cpp-python kullanacağız (GGUF için)
try:
    from llama_cpp import Llama
except ImportError:
    print("⚠️  llama-cpp-python yüklü değil. Kurulum:")
    print("   pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121")


@dataclass
class LLMConfig:
    """LLM konfigürasyon sınıfı"""
    name: str
    model_path: str
    context_length: int
    n_gpu_layers: int = -1  # -1 = tüm layerlar GPU'da
    n_ctx: int = 4096
    temperature: float = 0.7
    max_tokens: int = 512


class RAGSystem:
    """RAG (Retrieval-Augmented Generation) Sistemi"""
    
    def __init__(self, df: pd.DataFrame, index: faiss.Index, 
                 embeddings: np.ndarray, embedding_model):
        """
        Args:
            df: Veri DataFrame'i
            index: FAISS index
            embeddings: Document embeddings
            embedding_model: Query embedding için model
        """
        self.df = df
        self.index = index
        self.embeddings = embeddings
        self.embedding_model = embedding_model
        
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """
        Query için en alakalı dokümanları getir
        
        Args:
            query: Soru/sorgu
            k: Getirilecek doküman sayısı
            
        Returns:
            Retrieved documents listesi
        """
        # Query'yi embed et
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Arama yap
        distances, indices = self.index.search(query_embedding, k)
        
        # Sonuçları hazırla
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            doc = self.df.iloc[idx]
            results.append({
                'rank': i + 1,
                'similarity': float(dist),
                'title': doc['title'],
                'abstract': doc['abstract'],
                'category': doc.get('primary_category', 'unknown'),
                'index': int(idx)
            })
        
        return results
    
    def format_context(self, retrieved_docs: List[Dict], max_docs: int = 3) -> str:
        """
        Retrieved dokümanları context olarak formatla
        
        Args:
            retrieved_docs: Retrieve edilen dokümanlar
            max_docs: Kullanılacak maksimum doküman sayısı
        """
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs[:max_docs], 1):
            context_parts.append(
                f"[Document {i}]\n"
                f"Title: {doc['title']}\n"
                f"Abstract: {doc['abstract']}\n"
                f"Category: {doc['category']}\n"
            )
        
        return "\n".join(context_parts)


class LLMComparison:
    """LLM modellerini karşılaştırma sınıfı"""
    
    # Test edilecek modeller
    MODELS = {
        'llama-3.1-8b': LLMConfig(
            name='Llama-3.1-8B-Instruct',
            model_path='models/llama-3.1-8b-instruct.Q4_K_M.gguf',
            context_length=8192,
            n_ctx=4096
        ),
        'llama-3.2-3b': LLMConfig(
            name='Llama-3.2-3B-Instruct',
            model_path='models/llama-3.2-3b-instruct.Q4_K_M.gguf',
            context_length=4096,
            n_ctx=4096
        ),
        'mistral-7b': LLMConfig(
            name='Mistral-7B-Instruct-v0.3',
            model_path='models/mistral-7b-instruct-v0.3.Q4_K_M.gguf',
            context_length=8192,
            n_ctx=4096
        ),
        'phi-3-mini': LLMConfig(
            name='Phi-3-Mini-4K-Instruct',
            model_path='models/phi-3-mini-4k-instruct.Q4_K_M.gguf',
            context_length=4096,
            n_ctx=4096
        )
    }
    
    # User level promptları
    LEVEL_PROMPTS = {
        'beginner': """You are explaining to someone with NO technical background in this field. 
Use simple language, avoid jargon, and use everyday analogies. 
Assume the reader is intelligent but completely new to this topic.""",
        
        'intermediate': """You are explaining to someone with basic understanding of the field.
Use some technical terms but explain them when first introduced.
The reader has general science background but is not an expert in this specific area.""",
        
        'advanced': """You are explaining to someone with good understanding of the field.
Use technical terminology freely. The reader is familiar with fundamental concepts 
and is looking for deeper insights.""",
        
        'expert': """You are explaining to a fellow researcher in this field.
Use full technical detail, advanced terminology, and assume deep prior knowledge.
Focus on novel insights, methodological details, and implications."""
    }
    
    def __init__(self, rag_system: RAGSystem, output_dir: str = './llm_results'):
        """
        Args:
            rag_system: RAG sistemi
            output_dir: Sonuç kayıt dizini
        """
        self.rag = rag_system
        self.output_dir = output_dir
        self.results = {}
        
        os.makedirs(output_dir, exist_ok=True)
    
    def create_prompt(self, query: str, context: str, level: str = 'intermediate') -> str:
        """
        RAG prompt oluştur
        
        Args:
            query: Kullanıcı sorusu
            context: Retrieved dokümanlardan oluşan context
            level: Kullanıcı seviyesi (beginner/intermediate/advanced/expert)
        """
        level_instruction = self.LEVEL_PROMPTS.get(level, self.LEVEL_PROMPTS['intermediate'])
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant specialized in explaining scientific research papers.

{level_instruction}

Use the following research paper excerpts to answer the question. If the excerpts don't contain 
relevant information, say so and provide a general answer based on your knowledge.

<|eot_id|><|start_header_id|>user<|end_header_id|>

CONTEXT FROM RESEARCH PAPERS:
{context}

QUESTION: {query}

Please provide a clear, accurate answer appropriate for the user's level.

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return prompt
    
    def load_model(self, model_key: str) -> Llama:
        """
        LLM modelini yükle
        
        Args:
            model_key: Model anahtarı
        """
        config = self.MODELS[model_key]
        
        print(f"📥 Model yükleniyor: {config.name}")
        print(f"   Path: {config.model_path}")
        
        try:
            model = Llama(
                model_path=config.model_path,
                n_ctx=config.n_ctx,
                n_gpu_layers=config.n_gpu_layers,
                verbose=False
            )
            print(f"✅ Model yüklendi!")
            return model
            
        except FileNotFoundError:
            print(f"❌ Model dosyası bulunamadı: {config.model_path}")
            print(f"\n💡 İndirme önerisi:")
            print(f"   Hugging Face'den GGUF formatında indirin")
            print(f"   Örnek: https://huggingface.co/TheBloke")
            raise
    
    def generate_answer(self, model: Llama, prompt: str, 
                       max_tokens: int = 512) -> Dict:
        """
        LLM ile cevap üret
        
        Args:
            model: LLM modeli
            prompt: Prompt
            max_tokens: Maksimum token sayısı
        """
        start_time = time.time()
        
        try:
            output = model(
                prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                echo=False,
                stop=["<|eot_id|>", "<|end_of_text|>"]
            )
            
            generation_time = time.time() - start_time
            
            answer = output['choices'][0]['text'].strip()
            tokens_generated = output['usage']['completion_tokens']
            
            return {
                'answer': answer,
                'generation_time': generation_time,
                'tokens_generated': tokens_generated,
                'tokens_per_second': tokens_generated / generation_time if generation_time > 0 else 0
            }
            
        except Exception as e:
            print(f"❌ Generation hatası: {str(e)}")
            return {
                'answer': f"ERROR: {str(e)}",
                'generation_time': 0,
                'tokens_generated': 0,
                'tokens_per_second': 0
            }
    
    def evaluate_single_query(self, model_key: str, query: str, 
                             level: str = 'intermediate', k_retrieve: int = 5):
        """
        Tek bir query için model değerlendirmesi
        
        Args:
            model_key: Model anahtarı
            query: Soru
            level: Kullanıcı seviyesi
            k_retrieve: Retrieve edilecek doküman sayısı
        """
        # 1. Retrieve
        start_retrieve = time.time()
        retrieved_docs = self.rag.retrieve(query, k=k_retrieve)
        retrieve_time = time.time() - start_retrieve
        
        # 2. Context oluştur
        context = self.rag.format_context(retrieved_docs, max_docs=3)
        
        # 3. Prompt oluştur
        prompt = self.create_prompt(query, context, level)
        
        # 4. Model yükle (cache'lenebilir)
        model = self.load_model(model_key)
        
        # 5. Generate
        gen_result = self.generate_answer(model, prompt)
        
        return {
            'query': query,
            'level': level,
            'retrieved_docs': retrieved_docs,
            'retrieve_time': retrieve_time,
            'answer': gen_result['answer'],
            'generation_time': gen_result['generation_time'],
            'tokens_generated': gen_result['tokens_generated'],
            'tokens_per_second': gen_result['tokens_per_second'],
            'total_time': retrieve_time + gen_result['generation_time']
        }
    
    def run_test_queries(self, model_key: str, test_queries: List[Dict],
                        save_individual: bool = True):
        """
        Test query seti ile model değerlendirmesi
        
        Args:
            model_key: Model anahtarı
            test_queries: Test query listesi [{'query': ..., 'level': ...}, ...]
            save_individual: Her query sonucunu ayrı kaydet
        """
        print(f"\n{'='*80}")
        print(f"🚀 MODEL TEST: {model_key.upper()}")
        print(f"   Query sayısı: {len(test_queries)}")
        print(f"{'='*80}\n")
        
        model = self.load_model(model_key)
        
        results = []
        for i, test in enumerate(tqdm(test_queries, desc=f"Testing {model_key}")):
            query = test['query']
            level = test.get('level', 'intermediate')
            
            try:
                result = self.evaluate_single_query(
                    model_key, query, level, k_retrieve=5
                )
                results.append(result)
                
                # Individual save
                if save_individual:
                    individual_path = os.path.join(
                        self.output_dir,
                        f'{model_key}_query_{i+1}.json'
                    )
                    with open(individual_path, 'w') as f:
                        json.dump(result, f, indent=2)
                
            except Exception as e:
                print(f"❌ Query {i+1} hatası: {str(e)}")
                results.append({
                    'query': query,
                    'level': level,
                    'error': str(e)
                })
        
        # Aggregate results
        self.results[model_key] = {
            'model_config': self.MODELS[model_key].__dict__,
            'num_queries': len(test_queries),
            'successful_queries': len([r for r in results if 'error' not in r]),
            'avg_retrieve_time': np.mean([r.get('retrieve_time', 0) for r in results if 'error' not in r]),
            'avg_generation_time': np.mean([r.get('generation_time', 0) for r in results if 'error' not in r]),
            'avg_tokens_per_second': np.mean([r.get('tokens_per_second', 0) for r in results if 'error' not in r]),
            'total_time': np.sum([r.get('total_time', 0) for r in results if 'error' not in r]),
            'individual_results': results
        }
        
        # Save
        results_path = os.path.join(self.output_dir, f'{model_key}_results.json')
        with open(results_path, 'w') as f:
            # Convert non-serializable objects
            save_dict = self.results[model_key].copy()
            save_dict.pop('model_config', None)  # Remove non-serializable
            json.dump(save_dict, f, indent=2)
        
        print(f"\n✅ {model_key.upper()} testi tamamlandı!")
        print(f"   Başarılı: {self.results[model_key]['successful_queries']}/{len(test_queries)}")
        print(f"   Ort. Generation Time: {self.results[model_key]['avg_generation_time']:.2f}s")
        print(f"   Ort. Tokens/sec: {self.results[model_key]['avg_tokens_per_second']:.2f}")
        
        return results
    
    def create_comparison_report(self):
        """Karşılaştırma raporu"""
        if not self.results:
            print("❌ Henüz sonuç yok!")
            return
        
        print("\n" + "="*80)
        print("📊 LLM KARŞILAŞTIRMA RAPORU")
        print("="*80 + "\n")
        
        comparison_data = []
        for model_key, results in self.results.items():
            comparison_data.append({
                'Model': model_key.upper(),
                'Successful Queries': f"{results['successful_queries']}/{results['num_queries']}",
                'Avg Retrieve (s)': f"{results['avg_retrieve_time']:.3f}",
                'Avg Generation (s)': f"{results['avg_generation_time']:.2f}",
                'Avg Tokens/sec': f"{results['avg_tokens_per_second']:.2f}",
                'Total Time (s)': f"{results['total_time']:.2f}"
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        print(df_comparison.to_string(index=False))
        
        # Save
        csv_path = os.path.join(self.output_dir, 'llm_comparison.csv')
        df_comparison.to_csv(csv_path, index=False)
        print(f"\n💾 Karşılaştırma kaydedildi: {csv_path}")
        
        return df_comparison


# ============================================================================
# TEST QUERY SETİ OLUŞTURMA
# ============================================================================

def create_test_queries() -> List[Dict]:
    """Farklı seviyelerde test query'leri oluştur"""
    
    queries = [
        # Beginner level
        {'query': 'What is machine learning?', 'level': 'beginner'},
        {'query': 'How does artificial intelligence work?', 'level': 'beginner'},
        {'query': 'What is a neural network?', 'level': 'beginner'},
        
        # Intermediate level
        {'query': 'What is the difference between supervised and unsupervised learning?', 'level': 'intermediate'},
        {'query': 'How do transformers work in natural language processing?', 'level': 'intermediate'},
        {'query': 'What are the main challenges in computer vision?', 'level': 'intermediate'},
        
        # Advanced level
        {'query': 'What are the latest advances in reinforcement learning?', 'level': 'advanced'},
        {'query': 'How does attention mechanism improve transformer performance?', 'level': 'advanced'},
        {'query': 'What are the current limitations of large language models?', 'level': 'advanced'},
        
        # Expert level
        {'query': 'What are the theoretical foundations of gradient descent convergence?', 'level': 'expert'},
        {'query': 'How can we improve sample efficiency in off-policy RL algorithms?', 'level': 'expert'},
        {'query': 'What are the implications of the lottery ticket hypothesis?', 'level': 'expert'},
    ]
    
    return queries


# ============================================================================
# KULLANIM ÖRNEĞİ
# ============================================================================

if __name__ == "__main__":
    print("""
    🔧 KULLANIM ÖRNEĞİ:
    
    # 1. RAG sistemi hazırla
    import pickle
    from sentence_transformers import SentenceTransformer
    
    df = pd.read_pickle('/content/drive/MyDrive/arxiv_processed/arxiv_500k_processed.pkl')
    embeddings = np.load('/content/drive/MyDrive/arxiv_embeddings/embeddings_minilm.npy')
    index = faiss.read_index('/content/drive/MyDrive/arxiv_embeddings/faiss_index_minilm.index')
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    rag_system = RAGSystem(df, index, embeddings, embedding_model)
    
    # 2. LLM karşılaştırma
    llm_comp = LLMComparison(
        rag_system=rag_system,
        output_dir='/content/drive/MyDrive/llm_results'
    )
    
    # 3. Test query'leri oluştur
    test_queries = create_test_queries()
    
    # 4. Tek model test (önce bunu deneyin!)
    results = llm_comp.run_test_queries('llama-3.1-8b', test_queries)
    
    # 5. Rapor
    llm_comp.create_comparison_report()
    """)
