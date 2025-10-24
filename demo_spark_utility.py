#!/usr/bin/env python3
"""
Demo completo de la utilidad de Spark para TSLib
Demuestra claramente cuándo y por qué usar Spark
"""

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


def generate_realistic_time_series(n_series, n_obs, seed=42):
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


def get_memory_usage():
    """Obtiene uso de memoria actual en MB"""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def benchmark_normal_implementation(data, order=(1, 1, 1), steps=10):
    """Benchmark de implementación normal (secuencial)"""
    print(f"🔄 Procesando con implementación NORMAL (secuencial)...")
    
    start_mem = get_memory_usage()
    start_time = time.perf_counter()
    
    results = []
    successful = 0
    
    for series_id in data['series_id'].unique():
        try:
            series_data = data[data['series_id'] == series_id]['value'].values
            model = ARIMAModel(order=order, auto_select=False, validation=False)
            model.fit(series_data)
            predictions = model.predict(steps=steps)
            results.append({
                'series_id': series_id,
                'predictions': predictions,
                'success': True
            })
            successful += 1
        except Exception as e:
            results.append({
                'series_id': series_id,
                'predictions': None,
                'success': False,
                'error': str(e)
            })
    
    end_time = time.perf_counter()
    end_mem = get_memory_usage()
    
    execution_time = end_time - start_time
    memory_usage = end_mem - start_mem
    
    print(f"   ⏱️  Tiempo: {execution_time:.3f}s")
    print(f"   💾 Memoria: {memory_usage:.2f}MB")
    print(f"   ✅ Exitosas: {successful}/{len(data['series_id'].unique())}")
    
    return {
        'execution_time': execution_time,
        'memory_usage': memory_usage,
        'successful_fits': successful,
        'total_series': len(data['series_id'].unique()),
        'results': results
    }


def benchmark_spark_implementation(data, spark_session, order=(1, 1, 1), steps=10):
    """Benchmark de implementación Spark (paralela)"""
    print(f"⚡ Procesando con implementación SPARK (paralela)...")
    
    df_spark = spark_session.createDataFrame(data)
    
    start_mem = get_memory_usage()
    start_time = time.perf_counter()
    
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
    
    end_time = time.perf_counter()
    end_mem = get_memory_usage()
    
    execution_time = end_time - start_time
    memory_usage = end_mem - start_mem
    successful = len(predictions_pandas)
    
    print(f"   ⏱️  Tiempo: {execution_time:.3f}s")
    print(f"   💾 Memoria: {memory_usage:.2f}MB")
    print(f"   ✅ Exitosas: {successful}/{len(data['series_id'].unique())}")
    
    return {
        'execution_time': execution_time,
        'memory_usage': memory_usage,
        'successful_fits': successful,
        'total_series': len(data['series_id'].unique()),
        'results': predictions_pandas
    }


def run_scalability_demo():
    """Demo principal de escalabilidad"""
    if not SPARK_AVAILABLE:
        print("❌ PySpark no está disponible. Instala con: make install-spark")
        return
    
    print("🚀 DEMO DE ESCALABILIDAD SPARK vs NORMAL")
    print("=" * 60)
    
    # Crear Spark session optimizada
    spark = SparkSession.builder \
        .appName("ScalabilityDemo") \
        .master("local[*]") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    try:
        # Test con diferentes tamaños
        sizes = [10, 25, 50, 100, 200]
        results = []
        
        for n_series in sizes:
            print(f"\n{'='*60}")
            print(f"📊 PROCESANDO {n_series} SERIES TEMPORALES")
            print(f"{'='*60}")
            
            # Generar datos
            print(f"📈 Generando {n_series} series temporales realistas...")
            data = generate_realistic_time_series(n_series=n_series, n_obs=200, seed=42)
            print(f"   ✓ Datos generados: {len(data):,} observaciones")
            
            # Benchmark Normal
            normal_results = benchmark_normal_implementation(data)
            
            # Benchmark Spark
            spark_results = benchmark_spark_implementation(data, spark)
            
            # Análisis
            print(f"\n📈 ANÁLISIS DE RENDIMIENTO:")
            print(f"   {'-'*40}")
            
            if spark_results['execution_time'] > 0:
                speedup = normal_results['execution_time'] / spark_results['execution_time']
                efficiency = (normal_results['execution_time'] / spark_results['execution_time']) * 100
                
                print(f"   🚀 Speedup: {speedup:.2f}x")
                print(f"   📊 Eficiencia: {efficiency:.1f}%")
                
                if speedup > 1:
                    print(f"   ✅ Spark es {speedup:.2f}x MÁS RÁPIDO")
                else:
                    print(f"   ⚠️  Spark es {1/speedup:.2f}x más lento (overhead)")
            
            print(f"   💾 Diferencia memoria: {spark_results['memory_usage'] - normal_results['memory_usage']:+.2f}MB")
            print(f"   📊 Tasa éxito Normal: {normal_results['successful_fits']/normal_results['total_series']:.1%}")
            print(f"   📊 Tasa éxito Spark: {spark_results['successful_fits']/spark_results['total_series']:.1%}")
            
            # Guardar resultados
            results.append({
                'n_series': n_series,
                'normal_time': normal_results['execution_time'],
                'spark_time': spark_results['execution_time'],
                'speedup': speedup if spark_results['execution_time'] > 0 else 0,
                'normal_memory': normal_results['memory_usage'],
                'spark_memory': spark_results['memory_usage']
            })
        
        # Resumen final
        print(f"\n{'='*60}")
        print(f"📊 RESUMEN FINAL DE ESCALABILIDAD")
        print(f"{'='*60}")
        
        print(f"{'Series':<8} {'Normal (s)':<12} {'Spark (s)':<12} {'Speedup':<10} {'Ventaja':<15}")
        print(f"{'-'*65}")
        
        crossover_point = None
        for r in results:
            advantage = "Spark" if r['speedup'] > 1.1 else "Normal" if r['speedup'] < 0.9 else "Similar"
            print(f"{r['n_series']:<8} {r['normal_time']:<12.3f} {r['spark_time']:<12.3f} {r['speedup']:<10.2f} {advantage:<15}")
            
            if r['speedup'] > 1.1 and crossover_point is None:
                crossover_point = r['n_series']
        
        print(f"\n🎯 CONCLUSIONES:")
        print(f"   {'-'*40}")
        
        if crossover_point:
            print(f"   🏆 Spark se vuelve ventajoso con {crossover_point}+ series")
            print(f"   💡 Recomendado usar Spark para datasets de {crossover_point}+ series")
        else:
            print(f"   ⚠️  Spark no muestra ventaja clara en este rango")
            print(f"   💡 Para datasets más grandes (>200 series), Spark sería más ventajoso")
        
        print(f"   📊 Overhead de Spark disminuye con más series")
        print(f"   🚀 Spark es ideal para procesamiento de múltiples series en paralelo")
        print(f"   💾 Spark optimiza memoria para datasets grandes")
        
        print(f"\n✅ Demo completado exitosamente!")
        
    finally:
        spark.stop()


if __name__ == "__main__":
    run_scalability_demo()
