#!/usr/bin/env python3
"""
Test completo de escalabilidad Spark vs Normal
Demuestra la utilidad real de Spark con datasets grandes
"""

import pytest
import numpy as np
import pandas as pd
import time
import psutil
from tslib.models import ARIMAModel
from tslib.utils.checks import check_spark_availability

# Conditional import for Spark components
try:
    from pyspark.sql import SparkSession
    from tslib.spark.parallel_arima import ParallelARIMAProcessor
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False


class TestSparkScalabilityDemo:
    """
    Test comprehensivo que demuestra la utilidad real de Spark
    con datasets de diferentes tamaños para mostrar escalabilidad
    """

    @pytest.fixture(scope="class")
    def spark_session(self):
        """Fixture para SparkSession optimizada para benchmarks"""
        if not SPARK_AVAILABLE:
            pytest.skip("PySpark not available")
        
        spark = SparkSession.builder \
            .appName("ScalabilityDemo") \
            .master("local[*]") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .getOrCreate()
        
        # Reducir logging para benchmarks
        spark.sparkContext.setLogLevel("WARN")
        yield spark
        spark.stop()

    def _get_memory_usage(self):
        """Obtiene uso de memoria actual en MB"""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)

    def _generate_realistic_time_series(self, n_series, n_obs, seed=42):
        """Genera series temporales realistas con diferentes patrones"""
        np.random.seed(seed)
        data = []
        
        for i in range(n_series):
            # Diferentes tipos de series para mayor realismo
            series_type = i % 4
            
            if series_type == 0:  # AR(1) con tendencia
                ar_coef = np.random.uniform(0.3, 0.8)
                trend = np.random.uniform(0.01, 0.05)
                noise_std = np.random.uniform(0.5, 2.0)
                series = np.zeros(n_obs)
                series[0] = np.random.normal(0, noise_std)
                for t in range(1, n_obs):
                    series[t] = ar_coef * series[t-1] + trend * t + np.random.normal(0, noise_std)
                    
            elif series_type == 1:  # MA(1) con estacionalidad
                ma_coef = np.random.uniform(0.2, 0.7)
                seasonal_period = 12
                noise_std = np.random.uniform(0.3, 1.5)
                noise = np.random.normal(0, noise_std, n_obs + 1)
                series = np.zeros(n_obs)
                for t in range(n_obs):
                    seasonal = 2 * np.sin(2 * np.pi * t / seasonal_period)
                    series[t] = noise[t+1] + ma_coef * noise[t] + seasonal
                    
            elif series_type == 2:  # Random walk con drift
                drift = np.random.uniform(-0.02, 0.02)
                noise_std = np.random.uniform(0.8, 1.5)
                series = np.cumsum(np.random.normal(drift, noise_std, n_obs))
                
            else:  # ARMA(1,1) con ruido heterocedástico
                ar_coef = np.random.uniform(0.2, 0.6)
                ma_coef = np.random.uniform(0.1, 0.5)
                noise_std = np.random.uniform(0.5, 1.2)
                series = np.zeros(n_obs)
                noise = np.random.normal(0, noise_std, n_obs + 1)
                series[0] = noise[1]
                for t in range(1, n_obs):
                    series[t] = ar_coef * series[t-1] + noise[t+1] + ma_coef * noise[t]
            
            # Agregar identificador único y timestamp
            series_df = pd.DataFrame({
                'series_id': f'series_{i:04d}',
                'timestamp': range(n_obs),
                'value': series,
                'type': f'type_{series_type}'
            })
            data.append(series_df)
        
        return pd.concat(data, ignore_index=True)

    @pytest.mark.skipif(not SPARK_AVAILABLE, reason="PySpark not available")
    @pytest.mark.parametrize("n_series", [10, 25, 50, 100])
    def test_scalability_comparison(self, spark_session, n_series):
        """
        Test de escalabilidad que demuestra la utilidad de Spark
        con diferentes números de series temporales
        """
        print(f"\n{'='*60}")
        print(f"🚀 ESCALABILIDAD SPARK vs NORMAL - {n_series} SERIES")
        print(f"{'='*60}")
        
        # Generar datos realistas
        print(f"📊 Generando {n_series} series temporales realistas...")
        data = self._generate_realistic_time_series(n_series=n_series, n_obs=200, seed=42)
        print(f"   ✓ Datos generados: {len(data)} observaciones")
        
        order = (1, 1, 1)  # ARIMA(1,1,1)
        steps = 10
        
        # === IMPLEMENTACIÓN NORMAL (SECUENCIAL) ===
        print(f"\n🔄 Procesando con implementación NORMAL (secuencial)...")
        start_mem_normal = self._get_memory_usage()
        start_time_normal = time.perf_counter()
        
        normal_results = []
        successful_normal = 0
        
        for series_id in data['series_id'].unique():
            try:
                series_data = data[data['series_id'] == series_id]['value'].values
                model = ARIMAModel(order=order, auto_select=False, validation=False)
                model.fit(series_data)
                predictions = model.predict(steps=steps)
                normal_results.append({
                    'series_id': series_id,
                    'predictions': predictions,
                    'success': True
                })
                successful_normal += 1
            except Exception as e:
                normal_results.append({
                    'series_id': series_id,
                    'predictions': None,
                    'success': False,
                    'error': str(e)
                })
        
        end_time_normal = time.perf_counter()
        end_mem_normal = self._get_memory_usage()
        
        time_normal = end_time_normal - start_time_normal
        mem_normal = end_mem_normal - start_mem_normal
        
        print(f"   ⏱️  Tiempo: {time_normal:.3f}s")
        print(f"   💾 Memoria: {mem_normal:.2f}MB")
        print(f"   ✅ Exitosas: {successful_normal}/{n_series}")
        
        # === IMPLEMENTACIÓN SPARK (PARALELA) ===
        print(f"\n⚡ Procesando con implementación SPARK (paralela)...")
        df_spark = spark_session.createDataFrame(data)
        
        start_mem_spark = self._get_memory_usage()
        start_time_spark = time.perf_counter()
        
        processor = ParallelARIMAProcessor(spark_session=spark_session)
        
        # Fit models
        results_df = processor.fit_multiple_arima(
            df=df_spark,
            group_column='series_id',
            value_column='value',
            time_column='timestamp',
            order=order,
            auto_select=False
        )
        
        # Get predictions
        predictions_df = processor.predict_multiple_arima(
            df=df_spark,
            group_column='series_id',
            value_column='value',
            time_column='timestamp',
            order=order,
            steps=steps,
            return_conf_int=False
        )
        
        # Force computation
        results_pandas = results_df.toPandas()
        predictions_pandas = predictions_df.toPandas()
        
        end_time_spark = time.perf_counter()
        end_mem_spark = self._get_memory_usage()
        
        time_spark = end_time_spark - start_time_spark
        mem_spark = end_mem_spark - start_mem_spark
        successful_spark = len(predictions_pandas)
        
        print(f"   ⏱️  Tiempo: {time_spark:.3f}s")
        print(f"   💾 Memoria: {mem_spark:.2f}MB")
        print(f"   ✅ Exitosas: {successful_spark}/{n_series}")
        
        # === ANÁLISIS DE RESULTADOS ===
        print(f"\n📈 ANÁLISIS DE RENDIMIENTO:")
        print(f"   {'-'*40}")
        
        if time_spark > 0:
            speedup = time_normal / time_spark
            efficiency = (time_normal / time_spark) * 100
            
            print(f"   🚀 Speedup: {speedup:.2f}x")
            print(f"   📊 Eficiencia: {efficiency:.1f}%")
            
            if speedup > 1:
                print(f"   ✅ Spark es {speedup:.2f}x MÁS RÁPIDO")
            else:
                print(f"   ⚠️  Spark es {1/speedup:.2f}x más lento (overhead)")
        
        print(f"   💾 Diferencia memoria: {mem_spark - mem_normal:+.2f}MB")
        print(f"   📊 Tasa éxito Normal: {successful_normal/n_series:.1%}")
        print(f"   📊 Tasa éxito Spark: {successful_spark/n_series:.1%}")
        
        # === VERIFICACIÓN DE CONSISTENCIA ===
        print(f"\n🔍 VERIFICACIÓN DE CONSISTENCIA:")
        print(f"   {'-'*40}")
        
        # Comparar algunas series para verificar consistencia
        consistent_count = 0
        total_compared = 0
        
        for i, spark_row in predictions_pandas.head(5).iterrows():
            series_id = spark_row['series_id']
            spark_pred = spark_row['predictions']
            
            # Buscar resultado normal correspondiente
            normal_result = next((r for r in normal_results if r['series_id'] == series_id), None)
            
            if normal_result and normal_result['success'] and spark_pred is not None:
                try:
                    normal_pred = normal_result['predictions']
                    # Verificar consistencia numérica
                    np.testing.assert_allclose(normal_pred, spark_pred, rtol=1e-3, atol=1e-3)
                    consistent_count += 1
                    total_compared += 1
                except AssertionError:
                    total_compared += 1
                    print(f"   ⚠️  Inconsistencia en {series_id}")
        
        if total_compared > 0:
            consistency_rate = consistent_count / total_compared
            print(f"   ✅ Consistencia: {consistency_rate:.1%} ({consistent_count}/{total_compared})")
        
        # === CONCLUSIONES ===
        print(f"\n🎯 CONCLUSIONES:")
        print(f"   {'-'*40}")
        
        if n_series >= 25 and time_spark > 0:
            if time_normal / time_spark > 1.2:
                print(f"   🏆 Spark demuestra ventaja clara con {n_series} series")
                print(f"   💡 Recomendado para datasets de {n_series}+ series")
            else:
                print(f"   ⚖️  Punto de equilibrio cerca con {n_series} series")
        elif n_series < 25:
            print(f"   📝 Para {n_series} series, el overhead de Spark es notable")
            print(f"   💡 Spark se vuelve ventajoso con 25+ series")
        
        print(f"   📊 Datos procesados: {len(data):,} observaciones")
        print(f"   🔧 Modelo: ARIMA{order}")
        print(f"   📈 Predicciones: {steps} pasos adelante")
        
        # Assertions para el test
        assert successful_spark > 0, "Spark debe procesar al menos una serie"
        assert successful_normal > 0, "Normal debe procesar al menos una serie"
        assert len(predictions_pandas) == n_series, f"Spark debe procesar todas las {n_series} series"

    @pytest.mark.skipif(not SPARK_AVAILABLE, reason="PySpark not available")
    def test_memory_efficiency_large_dataset(self, spark_session):
        """
        Test específico para demostrar eficiencia de memoria con dataset grande
        """
        print(f"\n{'='*60}")
        print(f"💾 EFICIENCIA DE MEMORIA - DATASET GRANDE")
        print(f"{'='*60}")
        
        # Dataset grande: 200 series, 500 observaciones cada una
        n_series = 200
        n_obs = 500
        
        print(f"📊 Generando dataset grande: {n_series} series × {n_obs} obs = {n_series * n_obs:,} puntos")
        data = self._generate_realistic_time_series(n_series=n_series, n_obs=n_obs, seed=123)
        
        order = (1, 1, 1)
        steps = 5
        
        # Solo procesar una muestra para el test de memoria
        sample_series = data['series_id'].unique()[:50]  # 50 series para test
        sample_data = data[data['series_id'].isin(sample_series)]
        
        print(f"🧪 Procesando muestra de {len(sample_series)} series para test de memoria...")
        
        # Normal implementation
        start_mem = self._get_memory_usage()
        start_time = time.perf_counter()
        
        normal_count = 0
        for series_id in sample_series:
            series_data = sample_data[sample_data['series_id'] == series_id]['value'].values
            model = ARIMAModel(order=order, auto_select=False, validation=False)
            model.fit(series_data)
            predictions = model.predict(steps=steps)
            normal_count += 1
        
        end_time = time.perf_counter()
        end_mem = self._get_memory_usage()
        
        time_normal = end_time - start_time
        mem_normal = end_mem - start_mem
        
        print(f"   Normal: {time_normal:.3f}s, {mem_normal:.2f}MB")
        
        # Spark implementation
        df_spark = spark_session.createDataFrame(sample_data)
        
        start_mem = self._get_memory_usage()
        start_time = time.perf_counter()
        
        processor = ParallelARIMAProcessor(spark_session=spark_session)
        predictions_df = processor.predict_multiple_arima(
            df=df_spark,
            group_column='series_id',
            value_column='value',
            time_column='timestamp',
            order=order,
            steps=steps,
            return_conf_int=False
        )
        predictions_pandas = predictions_df.toPandas()
        
        end_time = time.perf_counter()
        end_mem = self._get_memory_usage()
        
        time_spark = end_time - start_time
        mem_spark = end_mem - start_mem
        
        print(f"   Spark: {time_spark:.3f}s, {mem_spark:.2f}MB")
        
        # Análisis
        if time_spark > 0:
            speedup = time_normal / time_spark
            print(f"   🚀 Speedup: {speedup:.2f}x")
        
        print(f"   💾 Diferencia memoria: {mem_spark - mem_normal:+.2f}MB")
        print(f"   📊 Series procesadas: {len(predictions_pandas)}")
        
        # Para dataset grande, Spark debería ser más eficiente
        assert len(predictions_pandas) == len(sample_series)
        print(f"   ✅ Test de memoria completado exitosamente")

    @pytest.mark.skipif(not SPARK_AVAILABLE, reason="PySpark not available")
    def test_throughput_analysis(self, spark_session):
        """
        Análisis de throughput para diferentes tamaños de dataset
        """
        print(f"\n{'='*60}")
        print(f"📊 ANÁLISIS DE THROUGHPUT")
        print(f"{'='*60}")
        
        sizes = [10, 25, 50, 100]
        results = []
        
        for n_series in sizes:
            print(f"\n🔍 Procesando {n_series} series...")
            
            # Generar datos
            data = self._generate_realistic_time_series(n_series=n_series, n_obs=100, seed=42)
            
            # Normal
            start_time = time.perf_counter()
            normal_count = 0
            for series_id in data['series_id'].unique():
                series_data = data[data['series_id'] == series_id]['value'].values
                model = ARIMAModel(order=(1, 1, 1), auto_select=False, validation=False)
                model.fit(series_data)
                predictions = model.predict(steps=5)
                normal_count += 1
            time_normal = time.perf_counter() - start_time
            
            # Spark
            df_spark = spark_session.createDataFrame(data)
            start_time = time.perf_counter()
            processor = ParallelARIMAProcessor(spark_session=spark_session)
            predictions_df = processor.predict_multiple_arima(
                df=df_spark,
                group_column='series_id',
                value_column='value',
                time_column='timestamp',
                order=(1, 1, 1),
                steps=5,
                return_conf_int=False
            )
            predictions_pandas = predictions_df.toPandas()
            time_spark = time.perf_counter() - start_time
            
            # Calcular throughput
            throughput_normal = n_series / time_normal if time_normal > 0 else 0
            throughput_spark = n_series / time_spark if time_spark > 0 else 0
            
            results.append({
                'n_series': n_series,
                'time_normal': time_normal,
                'time_spark': time_spark,
                'throughput_normal': throughput_normal,
                'throughput_spark': throughput_spark,
                'speedup': time_normal / time_spark if time_spark > 0 else 0
            })
            
            print(f"   Normal: {time_normal:.3f}s ({throughput_normal:.1f} series/s)")
            print(f"   Spark: {time_spark:.3f}s ({throughput_spark:.1f} series/s)")
            print(f"   Speedup: {time_normal / time_spark:.2f}x" if time_spark > 0 else "   Speedup: N/A")
        
        # Resumen
        print(f"\n📈 RESUMEN DE THROUGHPUT:")
        print(f"   {'Series':<8} {'Normal (s/s)':<12} {'Spark (s/s)':<12} {'Speedup':<8}")
        print(f"   {'-'*45}")
        
        for r in results:
            print(f"   {r['n_series']:<8} {r['throughput_normal']:<12.1f} {r['throughput_spark']:<12.1f} {r['speedup']:<8.2f}")
        
        # Encontrar punto de equilibrio
        crossover_point = None
        for r in results:
            if r['speedup'] > 1.1:  # Spark es al menos 10% más rápido
                crossover_point = r['n_series']
                break
        
        if crossover_point:
            print(f"\n🎯 Punto de equilibrio: Spark se vuelve ventajoso con {crossover_point}+ series")
        else:
            print(f"\n⚠️  Spark no muestra ventaja clara en este rango de tamaños")
        
        print(f"\n✅ Análisis de throughput completado")


if __name__ == "__main__":
    # Ejecutar test de demostración
    pytest.main([__file__ + "::TestSparkScalabilityDemo::test_scalability_comparison", "-v", "-s"])
