"""
PHASE 2: Embedding Model Comparison
Süre: ~40 saat (her model için ~10 saat)
Amaç: 4 farklı embedding modelini karşılaştırmak
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import time
from typing import List, Dict, Tuple
import json
import pickle
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class EmbeddingModelComparison:
    """
    Farklı embedding modellerini karşılaştırma sınıfı
    
    Modeller:
    1. all-MiniLM-L6-v2 (Hızlı, hafif - 384 dim)
    2. all-mpnet-base-v2 (Dengeli - 768 dim)
    3. bge-base-en-v1.5 (SOTA - 768 dim)
    4. multilingual-e5-base (Çok dilli - 768 dim)
    """
    
    MODELS = {
        'minilm': {
            'name': 'sentence-transformers/all-MiniLM-L6-v2',
            'dimension': 384,
            'description': 'Lightweight, fast model',
            'speed': 'Very Fast',
            'quality': 'Good'
        },
        'mpnet': {
            'name': 'sentence-transformers/all-mpnet-base-v2',
            'dimension': 768,
            'description': 'Balanced model',
            'speed': 'Medium',
            'quality': 'Very Good'
        },
        'bge': {
            'name': 'BAAI/bge-base-en-v1.5',
            'dimension': 768,
            'description': 'State-of-the-art model',
            'speed': 'Medium',
            'quality': 'Excellent'
        },
        'e5': {
            'name': 'intfloat/multilingual-e5-base',
            'dimension': 768,
            'description': 'Multilingual model',
            'speed': 'Medium',
            'quality': 'Very Good'
        }
    }
    
    def __init__(self, data_df: pd.DataFrame, output_dir: str = './embeddings'):
        """
        Args:
            data_df: İşlenmiş veri DataFrame'i
            output_dir: Embedding kayıt dizini
        """
        self.df = data_df
        self.output_dir = output_dir
        self.results = {}
        
        os.makedirs(output_dir, exist_ok=True)
        
    def create_embeddings(self, model_key: str, batch_size: int = 64):
        """
        Belirli bir model için embedding oluştur
        
        Args:
            model_key: Model anahtarı (minilm, mpnet, bge, e5)
            batch_size: Batch boyutu (A100 için 64-128 optimal)
        """
        print(f"\n{'='*80}")
        print(f"🚀 Model: {model_key.upper()}")
        print(f"   Açıklama: {self.MODELS[model_key]['description']}")
        print(f"   Model adı: {self.MODELS[model_key]['name']}")
        print(f"{'='*80}\n")
        
        # Model yükle
        print("📥 Model yükleniyor...")
        start_load = time.time()
        model = SentenceTransformer(self.MODELS[model_key]['name'])
        load_time = time.time() - start_load
        print(f"✅ Model yüklendi ({load_time:.2f} saniye)")
        
        # Metinleri hazırla
        texts = self.df['embedding_text'].tolist()
        print(f"📝 {len(texts):,} metin embedding'lenecek")
        
        # Embedding oluştur
        print(f"⚡ Embedding başlıyor (batch_size={batch_size})...")
        start_embed = time.time()
        
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # Cosine similarity için normalize
        )
        
        embed_time = time.time() - start_embed
        print(f"✅ Embedding tamamlandı ({embed_time:.2f} saniye)")
        print(f"   Hız: {len(texts)/embed_time:.2f} metin/saniye")
        
        # Kaydet
        save_path = os.path.join(
            self.output_dir, 
            f'embeddings_{model_key}.npy'
        )
        np.save(save_path, embeddings)
        print(f"💾 Embedding kaydedildi: {save_path}")
        
        # Sonuçları kaydet
        self.results[model_key] = {
            'model_name': self.MODELS[model_key]['name'],
            'dimension': embeddings.shape[1],
            'num_embeddings': len(embeddings),
            'load_time': load_time,
            'embed_time': embed_time,
            'speed_texts_per_sec': len(texts) / embed_time,
            'embedding_path': save_path,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return embeddings, self.results[model_key]
    
    def create_faiss_index(self, embeddings: np.ndarray, model_key: str, 
                          use_ivf: bool = True):
        """
        FAISS index oluştur
        
        Args:
            embeddings: Embedding matrisi
            model_key: Model anahtarı
            use_ivf: IVF (Inverted File) kullan (büyük veri için önerilir)
        """
        print(f"\n🔍 FAISS Index oluşturuluyor ({model_key})...")
        
        dimension = embeddings.shape[1]
        
        if use_ivf and len(embeddings) > 10000:
            # IVF index - daha hızlı arama için
            print(f"   • IVF Index kullanılıyor")
            nlist = min(int(np.sqrt(len(embeddings))), 1000)  # Cluster sayısı
            quantizer = faiss.IndexFlatIP(dimension)  # Inner Product (cosine için normalize edilmiş)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            
            print(f"   • Training... (nlist={nlist})")
            index.train(embeddings)
            
            print(f"   • Adding vectors...")
            index.add(embeddings)
            
            # Arama hassasiyeti
            index.nprobe = min(10, nlist)  # Araştırılacak cluster sayısı
            
        else:
            # Basit flat index
            print(f"   • Flat Index kullanılıyor")
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings)
        
        # Kaydet
        index_path = os.path.join(
            self.output_dir,
            f'faiss_index_{model_key}.index'
        )
        faiss.write_index(index, index_path)
        print(f"✅ Index kaydedildi: {index_path}")
        
        # Sonuçlara ekle
        self.results[model_key]['index_path'] = index_path
        self.results[model_key]['index_type'] = 'IVF' if use_ivf else 'Flat'
        
        return index
    
    def evaluate_retrieval(self, embeddings: np.ndarray, index: faiss.Index,
                          model_key: str, num_queries: int = 100, k: int = 5):
        """
        Retrieval performansını değerlendir
        
        Args:
            embeddings: Embedding matrisi
            index: FAISS index
            model_key: Model anahtarı
            num_queries: Test sorgu sayısı
            k: Top-K sonuç
        """
        print(f"\n📊 Retrieval değerlendirmesi ({model_key})...")
        
        # Rastgele query'ler seç
        np.random.seed(42)
        query_indices = np.random.choice(len(embeddings), num_queries, replace=False)
        query_embeddings = embeddings[query_indices]
        
        # Arama yap
        start_search = time.time()
        distances, indices = index.search(query_embeddings, k)
        search_time = time.time() - start_search
        
        # Metrikleri hesapla
        # Self-retrieval accuracy (ilk sonuç kendisi olmalı)
        self_retrieval_acc = np.mean(indices[:, 0] == query_indices)
        
        # Ortalama similarity
        avg_similarity = np.mean(distances)
        
        # Arama hızı
        queries_per_sec = num_queries / search_time
        
        print(f"   • Self-retrieval accuracy: {self_retrieval_acc:.4f}")
        print(f"   • Avg similarity (top-1): {avg_similarity:.4f}")
        print(f"   • Search speed: {queries_per_sec:.2f} queries/sec")
        
        # Sonuçlara ekle
        self.results[model_key].update({
            'self_retrieval_accuracy': float(self_retrieval_acc),
            'avg_similarity_top1': float(avg_similarity),
            'search_queries_per_sec': float(queries_per_sec),
            'eval_num_queries': num_queries,
            'eval_k': k
        })
        
        return {
            'self_retrieval_acc': self_retrieval_acc,
            'avg_similarity': avg_similarity,
            'queries_per_sec': queries_per_sec
        }
    
    def run_single_model(self, model_key: str, batch_size: int = 64):
        """Tek bir model için tüm pipeline'ı çalıştır"""
        print(f"\n{'#'*80}")
        print(f"# MODEL: {model_key.upper()}")
        print(f"{'#'*80}")
        
        # 1. Embedding oluştur
        embeddings, embed_results = self.create_embeddings(model_key, batch_size)
        
        # 2. FAISS index oluştur
        index = self.create_faiss_index(embeddings, model_key, use_ivf=True)
        
        # 3. Retrieval değerlendir
        eval_results = self.evaluate_retrieval(embeddings, index, model_key)
        
        print(f"\n✅ {model_key.upper()} tamamlandı!")
        
        return embeddings, index
    
    def run_all_models(self, batch_size: int = 64):
        """Tüm modelleri çalıştır"""
        print("\n" + "="*80)
        print("🚀 TÜM MODELLER İÇİN EMBEDDING BAŞLIYOR")
        print("="*80)
        
        all_embeddings = {}
        all_indices = {}
        
        for model_key in self.MODELS.keys():
            try:
                embeddings, index = self.run_single_model(model_key, batch_size)
                all_embeddings[model_key] = embeddings
                all_indices[model_key] = index
                
                # Her modelden sonra sonuçları kaydet (güvenlik için)
                self.save_results()
                
            except Exception as e:
                print(f"❌ HATA ({model_key}): {str(e)}")
                continue
        
        print("\n" + "="*80)
        print("✅ TÜM MODELLER TAMAMLANDI!")
        print("="*80)
        
        return all_embeddings, all_indices
    
    def save_results(self):
        """Sonuçları kaydet"""
        results_path = os.path.join(self.output_dir, 'embedding_comparison_results.json')
        
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Sonuçlar kaydedildi: {results_path}")
        
        return results_path
    
    def create_comparison_report(self):
        """Karşılaştırma raporu oluştur"""
        print("\n" + "="*80)
        print("📊 KARŞILAŞTIRMA RAPORU")
        print("="*80 + "\n")
        
        if not self.results:
            print("❌ Henüz sonuç yok!")
            return
        
        # Tablo oluştur
        comparison_data = []
        for model_key, results in self.results.items():
            comparison_data.append({
                'Model': model_key.upper(),
                'Dimension': results.get('dimension', 'N/A'),
                'Embed Time (s)': f"{results.get('embed_time', 0):.2f}",
                'Speed (texts/s)': f"{results.get('speed_texts_per_sec', 0):.2f}",
                'Self-Retrieval Acc': f"{results.get('self_retrieval_accuracy', 0):.4f}",
                'Avg Similarity': f"{results.get('avg_similarity_top1', 0):.4f}",
                'Search Speed (q/s)': f"{results.get('search_queries_per_sec', 0):.2f}"
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        print(df_comparison.to_string(index=False))
        
        # CSV olarak kaydet
        csv_path = os.path.join(self.output_dir, 'model_comparison.csv')
        df_comparison.to_csv(csv_path, index=False)
        print(f"\n💾 Karşılaştırma tablosu kaydedildi: {csv_path}")
        
        return df_comparison
    
    def visualize_comparison(self):
        """Karşılaştırmayı görselleştir"""
        if not self.results:
            print("❌ Henüz sonuç yok!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        models = list(self.results.keys())
        
        # 1. Embedding speed
        speeds = [self.results[m].get('speed_texts_per_sec', 0) for m in models]
        axes[0, 0].bar(models, speeds, color='skyblue', edgecolor='black')
        axes[0, 0].set_ylabel('Texts/Second')
        axes[0, 0].set_title('Embedding Speed Comparison')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. Self-retrieval accuracy
        accuracies = [self.results[m].get('self_retrieval_accuracy', 0) for m in models]
        axes[0, 1].bar(models, accuracies, color='lightgreen', edgecolor='black')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Self-Retrieval Accuracy')
        axes[0, 1].set_ylim([0, 1.0])
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. Average similarity
        similarities = [self.results[m].get('avg_similarity_top1', 0) for m in models]
        axes[1, 0].bar(models, similarities, color='lightcoral', edgecolor='black')
        axes[1, 0].set_ylabel('Similarity Score')
        axes[1, 0].set_title('Average Top-1 Similarity')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 4. Search speed
        search_speeds = [self.results[m].get('search_queries_per_sec', 0) for m in models]
        axes[1, 1].bar(models, search_speeds, color='lightyellow', edgecolor='black')
        axes[1, 1].set_ylabel('Queries/Second')
        axes[1, 1].set_title('Search Speed')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Kaydet
        viz_path = os.path.join(self.output_dir, 'model_comparison_visualization.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"\n💾 Görselleştirme kaydedildi: {viz_path}")
        
        plt.show()


# ============================================================================
# KULLANIM ÖRNEĞİ
# ============================================================================

if __name__ == "__main__":
    print("""
    🔧 KULLANIM ÖRNEĞİ:
    
    # Veri yükle
    df = pd.read_pickle('/content/drive/MyDrive/arxiv_processed/arxiv_500k_processed.pkl')
    
    # Karşılaştırma objesi oluştur
    comparison = EmbeddingModelComparison(
        data_df=df,
        output_dir='/content/drive/MyDrive/arxiv_embeddings'
    )
    
    # TEK MODEL TEST (önce bunu deneyin!)
    embeddings, index = comparison.run_single_model('minilm', batch_size=128)
    
    # TÜM MODELLERİ ÇALIŞTIR (uzun sürer!)
    all_embeddings, all_indices = comparison.run_all_models(batch_size=64)
    
    # Rapor ve görselleştirme
    comparison.create_comparison_report()
    comparison.visualize_comparison()
    """)
