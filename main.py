# main.py — Professional Stock Analyzer for TESLA & CROCS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import RobustScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings("ignore")

print("PROFESSIONAL STOCK ANALYZER: TESLA & CROCS")
print("=" * 60)

class StockAnalyzer:
    def __init__(self):
        self.stocks_data = {}
        self.current_analysis = {}
        
    def fetch_stock_data(self):
        """Загрузка данных по акциям"""
        print("🔄 Загружаем данные по акциям...")
        
        try:
            # Загружаем данные за последние 2 года
            tesla = yf.download("TSLA", start="2022-01-01", progress=False, auto_adjust=True)
            crocs = yf.download("CROX", start="2022-01-01", progress=False, auto_adjust=True)
            
            # Обрабатываем Tesla
            self.stocks_data['TSLA'] = self._process_stock_data(tesla, "Tesla Inc")
            
            # Обрабатываем Crocs
            self.stocks_data['CROX'] = self._process_stock_data(crocs, "Crocs Inc")
            
            print(f"✅ Загружены данные: Tesla ({len(tesla)} дней), Crocs ({len(crocs)} дней)")
            
            # Добавляем индикаторы для каждого stock
            for symbol in self.stocks_data:
                self._add_technical_indicators(symbol)
                self._add_market_indicators(symbol)
                self._add_sentiment_analysis(symbol)
            
            return True
            
        except Exception as e:
            print(f"❌ Ошибка загрузки данных: {e}")
            return False
    
    def _process_stock_data(self, stock_data, company_name):
        """Обработка данных по акциям"""
        df = pd.DataFrame()
        df['Date'] = stock_data.index
        df['Open'] = stock_data['Open'].values
        df['High'] = stock_data['High'].values
        df['Low'] = stock_data['Low'].values
        df['Close'] = stock_data['Close'].values
        df['Volume'] = stock_data['Volume'].values
        df['Company'] = company_name
        return df
    
    def _add_technical_indicators(self, symbol):
        """Добавление технических индикаторов"""
        df = self.stocks_data[symbol]
        
        try:
            # RSI
            def calculate_rsi(series, period=14):
                delta = series.diff()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                avg_gain = gain.ewm(com=period-1, adjust=False, min_periods=1).mean()
                avg_loss = loss.ewm(com=period-1, adjust=False, min_periods=1).mean()
                rs = avg_gain / avg_loss
                return 100 - (100 / (1 + rs))
            
            df['RSI_14'] = calculate_rsi(df['Close'], 14)
            
            # MACD
            ema_12 = df['Close'].ewm(span=12, adjust=False, min_periods=1).mean()
            ema_26 = df['Close'].ewm(span=26, adjust=False, min_periods=1).mean()
            df['MACD'] = ema_12 - ema_26
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # Bollinger Bands
            sma_20 = df['Close'].rolling(20, min_periods=1).mean()
            std_20 = df['Close'].rolling(20, min_periods=1).std()
            df['BB_Upper'] = sma_20 + (std_20 * 2)
            df['BB_Lower'] = sma_20 - (std_20 * 2)
            df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
            
            # Moving Averages
            df['SMA_20'] = df['Close'].rolling(20, min_periods=1).mean()
            df['SMA_50'] = df['Close'].rolling(50, min_periods=1).mean()
            df['SMA_200'] = df['Close'].rolling(200, min_periods=1).mean()
            df['EMA_21'] = df['Close'].ewm(span=21, adjust=False, min_periods=1).mean()
            
            # Support and Resistance
            df['Resistance'] = df['High'].rolling(20, min_periods=1).max()
            df['Support'] = df['Low'].rolling(20, min_periods=1).min()
            
        except Exception as e:
            print(f"⚠️ Ошибка в технических индикаторах для {symbol}: {e}")
    
    def _add_market_indicators(self, symbol):
        """Добавление рыночных индикаторов"""
        df = self.stocks_data[symbol]
        
        try:
            # Волатильность
            df['Volatility_7d'] = df['Close'].pct_change().rolling(7, min_periods=1).std()
            df['Volatility_30d'] = df['Close'].pct_change().rolling(30, min_periods=1).std()
            
            # Доходности
            df['Return_1d'] = df['Close'].pct_change(1)
            df['Return_7d'] = df['Close'].pct_change(7)
            df['Return_30d'] = df['Close'].pct_change(30)
            df['Return_YTD'] = (df['Close'] / df['Close'].iloc[0] - 1) * 100
            
            # Объем
            volume_sma_20 = df['Volume'].rolling(20, min_periods=1).mean()
            df['Volume_Ratio'] = df['Volume'] / volume_sma_20
            
            # Ценовые отношения
            df['High_Low_Ratio'] = df['High'] / df['Low']
            df['Close_Open_Ratio'] = df['Close'] / df['Open']
            
        except Exception as e:
            print(f"⚠️ Ошибка в рыночных индикаторах для {symbol}: {e}")
    
    def _add_sentiment_analysis(self, symbol):
        """Анализ настроения для каждой компании"""
        df = self.stocks_data[symbol]
        
        try:
            analyzer = SentimentIntensityAnalyzer()
            
            if symbol == 'TSLA':
                # Новости для Tesla
                news_items = [
                    "Tesla vehicle deliveries exceed analyst expectations",
                    "Elon Musk announces new AI and robotics initiatives",
                    "Tesla energy storage business shows strong growth",
                    "Cybertruck production ramps up successfully",
                    "Tesla faces competition from traditional automakers EVs",
                    "Autopilot and FSD technology advancements continue",
                    "Gigafactory expansions progress globally",
                    "Tesla battery technology improvements announced",
                    "Model 3 and Model Y maintain strong sales",
                    "Regulatory challenges for autonomous driving"
                ]
            else:  # CROX
                # Новости для Crocs
                news_items = [
                    "Crocs reports record quarterly revenue and profits",
                    "Celebrity collaborations drive brand popularity",
                    "International expansion shows strong results",
                    "Comfort footwear trend benefits Crocs sales",
                    "New product lines and designs launched successfully",
                    "E-commerce growth continues to accelerate",
                    "Sustainability initiatives in manufacturing",
                    "Partnerships with fashion brands increase appeal",
                    "Strong holiday season sales performance",
                    "Supply chain optimization improves margins"
                ]
            
            # Анализ тональности
            sentiment_scores = []
            for news in news_items:
                score = analyzer.polarity_scores(news)['compound']
                sentiment_scores.append(score)
            
            avg_sentiment = np.mean(sentiment_scores)
            df['News_Sentiment'] = np.clip(avg_sentiment, -0.5, 0.5)
            
        except Exception as e:
            print(f"⚠️ Ошибка анализа настроения для {symbol}: {e}")
            df['News_Sentiment'] = 0.0
    
    def analyze_stock(self, symbol):
        """Анализ конкретной акции"""
        if symbol not in self.stocks_data:
            return None
        
        df = self.stocks_data[symbol]
        current = df.iloc[-1]
        week_ago = df.iloc[-7] if len(df) >= 7 else df.iloc[0]
        month_ago = df.iloc[-30] if len(df) >= 30 else df.iloc[0]
        
        analysis = {
            'symbol': symbol,
            'company': current['Company'],
            'price': float(current['Close']),
            'price_change_7d': float((current['Close'] - week_ago['Close']) / week_ago['Close'] * 100),
            'price_change_30d': float((current['Close'] - month_ago['Close']) / month_ago['Close'] * 100),
            'volume_ratio': float(current['Volume_Ratio']),
            'rsi': float(current['RSI_14']),
            'sentiment': float(current['News_Sentiment']),
            'volatility_7d': float(current['Volatility_7d'] * 100),
            'macd_signal': 'БЫЧИЙ' if current['MACD_Histogram'] > 0 else 'МЕДВЕЖИЙ',
            'bb_position': float(current['BB_Position']),
            'trend_strength': self._calculate_trend_strength(symbol),
            'market_regime': self._identify_market_regime(symbol),
            'risk_level': self._assess_risk_level(symbol),
            'opportunity_score': self._calculate_opportunity_score(symbol),
            'support': float(current['Support']),
            'resistance': float(current['Resistance'])
        }
        
        return analysis
    
    def _calculate_trend_strength(self, symbol):
        """Расчет силы тренда"""
        try:
            df = self.stocks_data[symbol]
            sma_20 = df['SMA_20'].iloc[-1]
            sma_50 = df['SMA_50'].iloc[-1]
            current_price = df['Close'].iloc[-1]
            
            strength = 0
            
            # Краткосрочный тренд (20 дней)
            if current_price > sma_20:
                strength += 1
            else:
                strength -= 1
                
            # Среднесрочный тренд (50 дней)
            if current_price > sma_50:
                strength += 2
            else:
                strength -= 2
                
            # Долгосрочный тренд (200 дней)
            if 'SMA_200' in df.columns:
                sma_200 = df['SMA_200'].iloc[-1]
                if current_price > sma_200:
                    strength += 3
                else:
                    strength -= 3
            
            if strength >= 3:
                return "СИЛЬНЫЙ БЫЧИЙ"
            elif strength >= 1:
                return "УМЕРЕННЫЙ БЫЧИЙ"
            elif strength <= -3:
                return "СИЛЬНЫЙ МЕДВЕЖИЙ"
            elif strength <= -1:
                return "УМЕРЕННЫЙ МЕДВЕЖИЙ"
            else:
                return "БОКОВОЙ"
                
        except Exception as e:
            return "НЕОПРЕДЕЛЕН"
    
    def _identify_market_regime(self, symbol):
        """Идентификация рыночного режима"""
        try:
            df = self.stocks_data[symbol]
            volatility = df['Volatility_7d'].iloc[-1]
            rsi = df['RSI_14'].iloc[-1]
            volume_ratio = df['Volume_Ratio'].iloc[-1]
            
            if volatility > 0.04:  # Высокая волатильность для акций
                if volume_ratio > 1.5:
                    return "ВОЛАТИЛЬНЫЙ С ВЫСОКИМ ОБЪЕМОМ"
                else:
                    return "ВОЛАТИЛЬНЫЙ С НИЗКИМ ОБЪЕМОМ"
            elif rsi > 70:
                return "ПЕРЕКУПЛЕННОСТЬ"
            elif rsi < 30:
                return "ПЕРЕПРОДАННОСТЬ"
            elif volume_ratio > 1.2:
                return "АКТИВНЫЙ РОСТ"
            else:
                return "СТАБИЛЬНЫЙ"
                
        except Exception as e:
            return "НЕОПРЕДЕЛЕН"
    
    def _assess_risk_level(self, symbol):
        """Оценка уровня риска"""
        try:
            df = self.stocks_data[symbol]
            risk_score = 0
            
            # Волатильность
            vol_30d = df['Volatility_30d'].iloc[-1]
            if vol_30d > 0.05:
                risk_score += 3
            elif vol_30d > 0.03:
                risk_score += 2
            elif vol_30d > 0.02:
                risk_score += 1
            
            # RSI
            rsi = df['RSI_14'].iloc[-1]
            if rsi > 80 or rsi < 20:
                risk_score += 2
            elif rsi > 70 or rsi < 30:
                risk_score += 1
            
            if risk_score >= 4:
                return "ВЫСОКИЙ"
            elif risk_score >= 2:
                return "СРЕДНИЙ"
            else:
                return "НИЗКИЙ"
                
        except Exception as e:
            return "НЕОПРЕДЕЛЕН"
    
    def _calculate_opportunity_score(self, symbol):
        """Расчет оценки инвестиционных возможностей"""
        try:
            df = self.stocks_data[symbol]
            score = 50  # Нейтральная база
            
            # RSI
            rsi = df['RSI_14'].iloc[-1]
            if rsi < 30:
                score += 20  # Перепроданность - возможность покупки
            elif rsi > 70:
                score -= 20  # Перекупленность - возможность продажи
            
            # Тренд
            trend = self._calculate_trend_strength(symbol)
            if "БЫЧИЙ" in trend:
                score += 15
            elif "МЕДВЕЖИЙ" in trend:
                score -= 15
            
            # Настроение
            sentiment = df['News_Sentiment'].iloc[-1]
            score += int(sentiment * 10)
            
            return max(0, min(100, score))
            
        except Exception as e:
            return 50
    
    def generate_stock_recommendations(self, analysis):
        """Генерация рекомендаций для акции"""
        recommendations = []
        
        # Анализ RSI
        if analysis['rsi'] < 30:
            recommendations.append("📗 RSI показывает перепроданность - возможность покупки")
        elif analysis['rsi'] > 70:
            recommendations.append("📕 RSI показывает перекупленность - возможность продажи")
        
        # Анализ тренда
        if "БЫЧИЙ" in analysis['trend_strength']:
            recommendations.append("📈 Восходящий тренд - покупки на коррекциях")
        elif "МЕДВЕЖИЙ" in analysis['trend_strength']:
            recommendations.append("📉 Нисходящий тренд - продажи на отскоках")
        
        # Анализ поддержки/сопротивления
        current_price = analysis['price']
        support = analysis['support']
        resistance = analysis['resistance']
        
        if current_price <= support * 1.02:  # Вблизи поддержки
            recommendations.append("🛡️  Цена у уровня поддержки - возможность покупки")
        elif current_price >= resistance * 0.98:  # Вблизи сопротивления
            recommendations.append("🚧 Цена у уровня сопротивления - возможность продажи")
        
        # Общие рекомендации
        recommendations.append("⚡ Используйте стоп-лоссы для управления рисками")
        recommendations.append("📊 Рассмотрите диверсификацию портфеля")
        recommendations.append("🔍 Следите за квартальными отчетами компании")
        
        return recommendations
    
    def create_comparative_analysis(self):
        """Сравнительный анализ двух акций"""
        print("\n" + "="*80)
        print("📊 СРАВНИТЕЛЬНЫЙ АНАЛИЗ: TESLA vs CROCS")
        print("="*80)
        
        # Анализируем обе акции
        tsla_analysis = self.analyze_stock('TSLA')
        crox_analysis = self.analyze_stock('CROX')
        
        if not tsla_analysis or not crox_analysis:
            print("❌ Не удалось проанализировать одну из акций")
            return
        
        print(f"\n🏎️  TESLA INC (TSLA)")
        print(f"💰 Цена: ${tsla_analysis['price']:,.2f}")
        print(f"📈 Изменение за 7д: {tsla_analysis['price_change_7d']:+.2f}%")
        print(f"📊 RSI: {tsla_analysis['rsi']:.1f}")
        print(f"🎯 Оценка возможностей: {tsla_analysis['opportunity_score']}/100")
        print(f"⚡ Уровень риска: {tsla_analysis['risk_level']}")
        print(f"📈 Тренд: {tsla_analysis['trend_strength']}")
        
        print(f"\n👟 CROCS INC (CROX)")
        print(f"💰 Цена: ${crox_analysis['price']:,.2f}")
        print(f"📈 Изменение за 7д: {crox_analysis['price_change_7d']:+.2f}%")
        print(f"📊 RSI: {crox_analysis['rsi']:.1f}")
        print(f"🎯 Оценка возможностей: {crox_analysis['opportunity_score']}/100")
        print(f"⚡ Уровень риска: {crox_analysis['risk_level']}")
        print(f"📈 Тренд: {crox_analysis['trend_strength']}")
        
        # Сравнительная таблица
        print(f"\n📋 СРАВНИТЕЛЬНАЯ ТАБЛИЦА")
        print("-" * 60)
        print(f"{'Метрика':<25} {'TESLA':<15} {'CROCS':<15}")
        print("-" * 60)
        print(f"{'Цена ($)':<25} {tsla_analysis['price']:<15.2f} {crox_analysis['price']:<15.2f}")
        print(f"{'Изменение 7д (%)':<25} {tsla_analysis['price_change_7d']:<15.2f} {crox_analysis['price_change_7d']:<15.2f}")
        print(f"{'RSI':<25} {tsla_analysis['rsi']:<15.1f} {crox_analysis['rsi']:<15.1f}")
        print(f"{'Оценка возможностей':<25} {tsla_analysis['opportunity_score']:<15} {crox_analysis['opportunity_score']:<15}")
        print(f"{'Уровень риска':<25} {tsla_analysis['risk_level']:<15} {crox_analysis['risk_level']:<15}")
        print(f"{'Режим рынка':<25} {tsla_analysis['market_regime']:<15} {crox_analysis['market_regime']:<15}")
        
        # Рекомендации для Tesla
        print(f"\n💡 РЕКОМЕНДАЦИИ ДЛЯ TESLA:")
        tsla_recs = self.generate_stock_recommendations(tsla_analysis)
        for rec in tsla_recs:
            print(f"   • {rec}")
        
        # Рекомендации для Crocs
        print(f"\n💡 РЕКОМЕНДАЦИИ ДЛЯ CROCS:")
        crox_recs = self.generate_stock_recommendations(crox_analysis)
        for rec in crox_recs:
            print(f"   • {rec}")
        
        print("\n" + "="*80)
        print("⚠️  ВНИМАНИЕ: Это инструмент анализа, а не финансовый совет")
        print("💼 Консультируйтесь с финансовыми советниками перед инвестициями")
        print("="*80)
    
    def create_visualizations(self):
        """Создание визуализаций для обеих акций"""
        print("\n📈 Создаем сравнительные визуализации...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('СРАВНИТЕЛЬНЫЙ АНАЛИЗ: TESLA vs CROCS', fontsize=16, fontweight='bold')
        
        # График 1: Цены (нормализованные)
        ax1 = axes[0, 0]
        for symbol in ['TSLA', 'CROX']:
            df = self.stocks_data[symbol]
            # Нормализуем цены к 100 для сравнения
            normalized_price = (df['Close'].tail(100) / df['Close'].tail(100).iloc[0]) * 100
            ax1.plot(df['Date'].tail(100), normalized_price, label=symbol, linewidth=2)
        ax1.set_title('Сравнение динамики цен (нормализовано)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # График 2: RSI сравнение
        ax2 = axes[0, 1]
        for symbol in ['TSLA', 'CROX']:
            df = self.stocks_data[symbol]
            ax2.plot(df['Date'].tail(100), df['RSI_14'].tail(100), label=symbol, linewidth=2)
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Перекупленность')
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Перепроданность')
        ax2.set_title('Сравнение RSI')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # График 3: Объемы торгов
        ax3 = axes[0, 2]
        for symbol in ['TSLA', 'CROX']:
            df = self.stocks_data[symbol]
            # Нормализуем объемы для сравнения
            normalized_volume = df['Volume'].tail(50) / df['Volume'].tail(50).max()
            ax3.bar(df['Date'].tail(50), normalized_volume, alpha=0.7, label=symbol)
        ax3.set_title('Сравнение объемов (нормализовано)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        
        # График 4: Волатильность
        ax4 = axes[1, 0]
        for symbol in ['TSLA', 'CROX']:
            df = self.stocks_data[symbol]
            ax4.plot(df['Date'].tail(100), df['Volatility_7d'].tail(100) * 100, 
                    label=symbol, linewidth=2)
        ax4.set_title('Волатильность 7д (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
        
        # График 5: Отношение объемов
        ax5 = axes[1, 1]
        for symbol in ['TSLA', 'CROX']:
            df = self.stocks_data[symbol]
            ax5.plot(df['Date'].tail(100), df['Volume_Ratio'].tail(100), 
                    label=symbol, linewidth=2)
        ax5.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
        ax5.set_title('Отношение объема к среднему')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.tick_params(axis='x', rotation=45)
        
        # График 6: MACD гистограмма
        ax6 = axes[1, 2]
        for symbol in ['TSLA', 'CROX']:
            df = self.stocks_data[symbol]
            ax6.bar(df['Date'].tail(50), df['MACD_Histogram'].tail(50), 
                   alpha=0.6, label=symbol)
        ax6.axhline(y=0, color='black', linestyle='-', alpha=0.8)
        ax6.set_title('MACD Histogram сравнение')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

# Запуск анализатора
if __name__ == "__main__":
    try:
        analyzer = StockAnalyzer()
        success = analyzer.fetch_stock_data()
        
        if success:
            analyzer.create_comparative_analysis()
            analyzer.create_visualizations()
        else:
            print("❌ Не удалось загрузить данные для анализа")
            
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()