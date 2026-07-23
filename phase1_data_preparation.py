"""
PHASE 1: Data Preparation and Initial Embedding
Süre: ~30 saat
Amaç: 500K veri seçimi, temizleme ve ilk embedding oluşturma
"""

import pandas as pd
import numpy as np
import json
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pickle
import os
from datetime import datetime

class ArxivDataPreparation:
    """500K veri için optimize edilmiş veri hazırlama sınıfı"""
    
    def __init__(self, data_path: str, target_size: int = 500000):
        """
        Args:
            data_path: ArXiv JSON dosya yolu
            target_size: Hedef veri sayısı
        """
        self.data_path = data_path
        self.target_size = target_size
        self.df = None
        self.sampled_df = None
        
    def load_data_streaming(self, max_rows: int = None):
        """
        Streaming ile veri yükleme - Memory efficient
        
        Args:
            max_rows: Maksimum yüklenecek satır (None = hepsi)
        """
        print(f"📂 Veri yükleniyor (streaming mode)...")
        
        data_list = []
        with open(self.data_path, 'r') as f:
            for i, line in enumerate(f):
                if max_rows and i >= max_rows:
                    break
                    
                if i % 100000 == 0:
                    print(f"   ✓ {i:,} satır yüklendi...")
                
                try:
                    data_list.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        
        self.df = pd.DataFrame(data_list)
        print(f"✅ Toplam {len(self.df):,} satır yüklendi")
        return self.df
    
    def clean_data(self):
        """Veri temizleme ve ön işleme"""
        print("\n🧹 Veri temizleniyor...")
        
        initial_size = len(self.df)
        
        # 1. Boş değerleri temizle
        print("   • Boş değerler kontrol ediliyor...")
        self.df = self.df.dropna(subset=['title', 'abstract'])
        
        # 2. Duplicate kontrolü
        print("   • Duplicate'ler kontrol ediliyor...")
        self.df = self.df.drop_duplicates(subset=['id'])
        
        # 3. Minimum uzunluk kontrolü
        print("   • Minimum uzunluk kontrolü...")
        self.df = self.df[self.df['abstract'].str.len() > 100]
        
        # 4. Kategorileri parse et
        print("   • Kategoriler işleniyor...")
        if 'categories' in self.df.columns:
            # String olarak saklanmış kategorileri listeye çevir
            self.df['category_list'] = self.df['categories'].apply(
                lambda x: x.split() if isinstance(x, str) else []
            )
            self.df['primary_category'] = self.df['category_list'].apply(
                lambda x: x[0] if len(x) > 0 else 'unknown'
            )
        
        # 5. Tarih parse et
        print("   • Tarihler işleniyor...")
        if 'update_date' in self.df.columns:
            self.df['year'] = pd.to_datetime(
                self.df['update_date'], 
                errors='coerce'
            ).dt.year
        
        # 6. Metin birleştirme (embedding için)
        print("   • Embedding metinleri hazırlanıyor...")
        self.df['embedding_text'] = (
            "Title: " + self.df['title'] + 
            "\nAbstract: " + self.df['abstract']
        )
        
        removed = initial_size - len(self.df)
        print(f"✅ Temizleme tamamlandı: {removed:,} satır kaldırıldı")
        print(f"   Kalan veri: {len(self.df):,}")
        
        return self.df
    
    def stratified_sampling(self):
        """
        Kategorilere göre dengeli örnekleme
        """
        print(f"\n🎯 Stratified sampling yapılıyor ({self.target_size:,} hedef)...")
        
        # Kategori dağılımını hesapla
        category_counts = self.df['primary_category'].value_counts()
        print(f"   • Toplam {len(category_counts)} farklı kategori bulundu")
        
        # Her kategoriden proportional olarak örnekle
        sampling_ratio = self.target_size / len(self.df)
        
        sampled_dfs = []
        for category, count in category_counts.items():
            category_df = self.df[self.df['primary_category'] == category]
            sample_size = min(
                int(count * sampling_ratio),
                len(category_df)
            )
            
            if sample_size > 0:
                sampled = category_df.sample(n=sample_size, random_state=42)
                sampled_dfs.append(sampled)
        
        self.sampled_df = pd.concat(sampled_dfs, ignore_index=True)
        
        # Eğer hedeften az ise, rastgele ekle
        if len(self.sampled_df) < self.target_size:
            remaining = self.target_size - len(self.sampled_df)
            extra = self.df[~self.df.index.isin(self.sampled_df.index)].sample(
                n=min(remaining, len(self.df) - len(self.sampled_df)),
                random_state=42
            )
            self.sampled_df = pd.concat([self.sampled_df, extra], ignore_index=True)
        
        # Shuffle
        self.sampled_df = self.sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"✅ Sampling tamamlandı: {len(self.sampled_df):,} veri seçildi")
        return self.sampled_df
    
    def analyze_distribution(self):
        """Veri dağılımını analiz et ve görselleştir"""
        print("\n📊 Veri dağılımı analiz ediliyor...")
        
        stats = {
            'total_samples': len(self.sampled_df),
            'unique_categories': self.sampled_df['primary_category'].nunique(),
            'category_distribution': self.sampled_df['primary_category'].value_counts().to_dict(),
            'year_distribution': self.sampled_df['year'].value_counts().to_dict() if 'year' in self.sampled_df.columns else {},
            'avg_abstract_length': self.sampled_df['abstract'].str.len().mean(),
            'avg_title_length': self.sampled_df['title'].str.len().mean(),
        }
        
        # Top 20 kategoriyi göster
        print(f"\n📈 Top 20 Kategori:")
        top_categories = self.sampled_df['primary_category'].value_counts().head(20)
        for cat, count in top_categories.items():
            percentage = (count / len(self.sampled_df)) * 100
            print(f"   {cat:20s}: {count:6,} ({percentage:5.2f}%)")
        
        # Yıl dağılımı
        if 'year' in self.sampled_df.columns:
            print(f"\n📅 Yıl Dağılımı:")
            year_dist = self.sampled_df['year'].value_counts().sort_index().tail(10)
            for year, count in year_dist.items():
                print(f"   {year}: {count:,}")
        
        return stats
    
    def create_visualization(self, save_path: str = None):
        """Veri dağılımını görselleştir"""
        print("\n📊 Görselleştirmeler oluşturuluyor...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Top 20 kategori
        top_20_cats = self.sampled_df['primary_category'].value_counts().head(20)
        axes[0, 0].barh(range(len(top_20_cats)), top_20_cats.values)
        axes[0, 0].set_yticks(range(len(top_20_cats)))
        axes[0, 0].set_yticklabels(top_20_cats.index)
        axes[0, 0].set_xlabel('Count')
        axes[0, 0].set_title('Top 20 Categories Distribution')
        axes[0, 0].invert_yaxis()
        
        # 2. Yıl dağılımı
        if 'year' in self.sampled_df.columns:
            year_dist = self.sampled_df['year'].value_counts().sort_index()
            axes[0, 1].plot(year_dist.index, year_dist.values, marker='o')
            axes[0, 1].set_xlabel('Year')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].set_title('Papers by Year')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Abstract uzunluk dağılımı
        abstract_lengths = self.sampled_df['abstract'].str.len()
        axes[1, 0].hist(abstract_lengths, bins=50, edgecolor='black')
        axes[1, 0].set_xlabel('Abstract Length (characters)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Abstract Length Distribution')
        axes[1, 0].axvline(abstract_lengths.mean(), color='red', linestyle='--', 
                          label=f'Mean: {abstract_lengths.mean():.0f}')
        axes[1, 0].legend()
        
        # 4. Title uzunluk dağılımı
        title_lengths = self.sampled_df['title'].str.len()
        axes[1, 1].hist(title_lengths, bins=50, edgecolor='black', color='green', alpha=0.7)
        axes[1, 1].set_xlabel('Title Length (characters)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Title Length Distribution')
        axes[1, 1].axvline(title_lengths.mean(), color='red', linestyle='--',
                          label=f'Mean: {title_lengths.mean():.0f}')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Görselleştirme kaydedildi: {save_path}")
        
        plt.show()
    
    def save_processed_data(self, output_dir: str = './processed_data'):
        """İşlenmiş veriyi kaydet"""
        print(f"\n💾 Veri kaydediliyor: {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. CSV olarak kaydet
        csv_path = os.path.join(output_dir, 'arxiv_500k_processed.csv')
        self.sampled_df.to_csv(csv_path, index=False)
        print(f"   ✓ CSV kaydedildi: {csv_path}")
        
        # 2. Pickle olarak kaydet (daha hızlı yükleme)
        pickle_path = os.path.join(output_dir, 'arxiv_500k_processed.pkl')
        self.sampled_df.to_pickle(pickle_path)
        print(f"   ✓ Pickle kaydedildi: {pickle_path}")
        
        # 3. Metadata kaydet
        metadata = {
            'creation_date': datetime.now().isoformat(),
            'total_samples': len(self.sampled_df),
            'columns': list(self.sampled_df.columns),
            'primary_categories': self.sampled_df['primary_category'].unique().tolist(),
            'statistics': {
                'avg_abstract_length': float(self.sampled_df['abstract'].str.len().mean()),
                'avg_title_length': float(self.sampled_df['title'].str.len().mean()),
                'year_range': [
                    int(self.sampled_df['year'].min()) if 'year' in self.sampled_df.columns else None,
                    int(self.sampled_df['year'].max()) if 'year' in self.sampled_df.columns else None
                ]
            }
        }
        
        metadata_path = os.path.join(output_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"   ✓ Metadata kaydedildi: {metadata_path}")
        
        print(f"✅ Tüm veriler kaydedildi!")
        
        return {
            'csv_path': csv_path,
            'pickle_path': pickle_path,
            'metadata_path': metadata_path
        }
    
    def run_complete_pipeline(self, output_dir: str = './processed_data'):
        """Tüm pipeline'ı çalıştır"""
        print("="*80)
        print("🚀 PHASE 1: VERİ HAZIRLAMA BAŞLIYOR")
        print("="*80)
        
        # 1. Veri yükleme
        self.load_data_streaming(max_rows=2100000)  # 2.1M limit
        
        # 2. Veri temizleme
        self.clean_data()
        
        # 3. Stratified sampling
        self.stratified_sampling()
        
        # 4. Analiz
        stats = self.analyze_distribution()
        
        # 5. Görselleştirme
        viz_path = os.path.join(output_dir, 'data_distribution.png')
        os.makedirs(output_dir, exist_ok=True)
        self.create_visualization(save_path=viz_path)
        
        # 6. Kaydet
        paths = self.save_processed_data(output_dir)
        
        print("\n" + "="*80)
        print("✅ PHASE 1 TAMAMLANDI!")
        print("="*80)
        
        return self.sampled_df, stats, paths


# ============================================================================
# KULLANIM ÖRNEĞİ
# ============================================================================

if __name__ == "__main__":
    # Kullanım örneği
    print("""
    🔧 KULLANIM ÖRNEĞİ:
    
    # Colab'da:
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Veri hazırlama
    prep = ArxivDataPreparation(
        data_path='/content/drive/MyDrive/arxiv-metadata-oai-snapshot.json',
        target_size=500000
    )
    
    # Pipeline'ı çalıştır
    df, stats, paths = prep.run_complete_pipeline(
        output_dir='/content/drive/MyDrive/arxiv_processed'
    )
    
    print(f"İşlenmiş veri: {paths['pickle_path']}")
    """)
