import numpy as np, pandas as pd, traceback, engine

out = []
try:
    # 1) modelo actual (semilla)
    engine.ENGINE.bootstrap()
    b0 = engine.ENGINE.bundle
    coef0 = {c['feature']: c['coef'] for c in b0['learning'].get('logit_coef', [])}
    samp = {'edad':35,'ingresos_mensuales':1200,'cargas_familiares':1,'creditos_activos':1,
            'sexo':'Masculino','educacion':'Universitaria','historial_pagos':'Bueno',
            'institucion':'Banco Pichincha','tipo_credito':'Personal','situacion_laboral':'Empleado Privado'}
    s0 = engine.ENGINE.score(samp)['percent']
    out.append(f"SEMILLA: source={b0['source']} size={b0['dataset_size']} score={s0}% NNauc={b0['metrics']['neural_net']['auc']}")

    # 2) dataset 'real' DIFERENTE: ingresos altos penalizan (invertido) + institución nueva
    rng = np.random.default_rng(7); rows = []
    INST = engine.INSTITUTIONS + ['Cooperativa Real XYZ']  # categoría NUEVA no vista en la semilla
    for _ in range(1200):
        ing = float(max(0, rng.normal(900, 400))); car = int(rng.integers(0, 6))
        hp = rng.choice(engine.PAYMENT_HISTORY)
        z = 0.5 - (ing/1500)*1.4 + {'Malo':1.2,'Regular':0.3,'Bueno':-0.5,'Excelente':-1.3}[hp] - 0.1*car
        ap = int(rng.random() < 1/(1+np.exp(-z)))
        rows.append({'edad':int(rng.integers(20,70)),'ingresos_mensuales':round(ing,2),
            'cargas_familiares':car,'creditos_activos':int(rng.integers(0,6)),
            'sexo':rng.choice(['Masculino','Femenino']),'educacion':rng.choice(engine.EDUCATION),
            'historial_pagos':hp,'institucion':rng.choice(INST),'tipo_credito':rng.choice(engine.CREDIT_TYPES),
            'situacion_laboral':rng.choice(engine.EMPLOYMENT),'aprobado':ap})
    df = pd.DataFrame(rows)
    b1 = engine.retrain_from_dataframe(df, source='real_test.csv')
    coef1 = {c['feature']: c['coef'] for c in b1['learning'].get('logit_coef', [])}
    s1 = engine.ENGINE.score(samp)['percent']
    out.append(f"REAL:    source={b1['source']} size={b1['dataset_size']} score={s1}% NNauc={b1['metrics']['neural_net']['auc']} nfeat={b1['n_features']}")
    out.append(f"hot-reload (ENGINE usa el nuevo bundle): {engine.ENGINE.bundle is b1}")
    out.append(f"coef ingresos: semilla={coef0.get('ingresos_mensuales')} real={coef1.get('ingresos_mensuales')} -> CAMBIO={'SI' if coef0.get('ingresos_mensuales')!=coef1.get('ingresos_mensuales') else 'NO'}")
    out.append(f"score: {s0}% -> {s1}% -> CAMBIO={'SI' if s0!=s1 else 'NO'}")
    dn = b1['learning']['dead_neurons']
    out.append(f"red GELU reentrenada: loss_pts={len(b1['learning'].get('nn_loss_curve',[]))} dead={dn['total_dead']}/{dn['total_units']} sana={dn['healthy']}")
    out.append(f"institución NUEVA aprendida (columna creada): {'institucion_Cooperativa Real XYZ' in b1['columns']}")

    # 3) restaurar el modelo semilla (no dejar el de prueba persistido)
    bseed = engine.train_and_persist()
    engine.ENGINE.bundle = bseed
    out.append(f"RESTAURADO a semilla: source={bseed['source']} size={bseed['dataset_size']}")
except Exception:
    out.append("ERROR:\n" + traceback.format_exc())

open("verify_retrain_out.txt", "w", encoding="utf-8").write("\n".join(out))
print("\n".join(out))
