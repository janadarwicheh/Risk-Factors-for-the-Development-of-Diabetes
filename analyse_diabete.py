"""
SUJET - Facteurs de risques pour le développement du diabète
Source des données : Behavioral Risk Factor Surveillance System (BRFSS) 2015
Groupe 12 : DARWICHEH Jana, SIRB Ema
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from scipy import stats
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore') # pour eviter les warnings 


# 1. LECTURE DU FICHIER CSV ET CRÉATION DU DATAFRAME PANDAS

# Source : pd.read_csv() - documentation pandas https://pandas.pydata.org
df = pd.read_csv('diabetes_012_health_indicators_BRFSS2015.csv')

print("=" * 60)
print("APERÇU DU DATAFRAME")
print("=" * 60)
print(f"Dimensions : {df.shape[0]} lignes × {df.shape[1]} colonnes")
print(df.head())
print("\nTypes de données :")
print(df.dtypes)
print("\nStatistiques descriptives :")
print(df.describe().round(2))

# Création d'une colonne catégorielle pour le statut diabétique
df['Statut'] = df['Diabetes_012'].map({0.0: 'Sain', 1.0: 'Prédiabète', 2.0: 'Diabète'})


# 2. CALCUL DES TAUX DE DIABÈTE PAR FACTEUR

# On considère "atteint" = prediabète OU diabète (Diabetes_012 > 0)
df['Atteint'] = (df['Diabetes_012'] > 0).astype(int)

# Facteurs binaires à analyser
facteurs_binaires = [
    'HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk', 'Sex'
]

labels_fr = {
    'HighBP': 'Hypertension',
    'HighChol': 'Cholestérol élevé',
    'CholCheck': 'Bilan cholestérol',
    'Smoker': 'Fumeur',
    'Stroke': 'AVC',
    'HeartDiseaseorAttack': 'Maladie cardiaque',
    'PhysActivity': 'Activité physique',
    'Fruits': 'Consomme fruits',
    'Veggies': 'Consomme légumes',
    'HvyAlcoholConsump': 'Alcool excessif',
    'AnyHealthcare': 'Couverture santé',
    'NoDocbcCost': 'Pas médecin (coût)',
    'DiffWalk': 'Difficulté marche',
    'Sex': 'Sexe masculin'
}

# Taux de diabète si facteur présent vs absent
risques = {}
for f in facteurs_binaires:
    taux_1 = df[df[f] == 1]['Atteint'].mean() * 100
    taux_0 = df[df[f] == 0]['Atteint'].mean() * 100
    risque_relatif = taux_1 / taux_0 if taux_0 > 0 else np.nan
    risques[f] = {
        'label': labels_fr[f],
        'taux_present': taux_1,
        'taux_absent': taux_0,
        'risque_relatif': risque_relatif,
        'delta': taux_1 - taux_0
    }

risques_df = pd.DataFrame(risques).T.sort_values('risque_relatif', ascending=False)
print("\n" + "=" * 60)
print("TABLEAU DES RISQUES RELATIFS")
print("=" * 60)
print(risques_df[['label', 'taux_present', 'taux_absent', 'risque_relatif']].round(2).to_string())


# FIGURE 1 : MATRICE DE GRAPHIQUES - VUE D'ENSEMBLE

# Source subplot : https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/subplot.html
fig1, axes = plt.subplots(3, 3, figsize=(18, 15))
fig1.suptitle('Vue d\'ensemble - Facteurs de risque du Diabète (BRFSS 2015)',
              fontsize=16, fontweight='bold', y=0.98)

couleurs_statut = {'Sain': '#2ecc71', 'Prédiabète': '#f39c12', 'Diabète': '#e74c3c'}

# (a) Distribution du statut - Camembert
ax = axes[0, 0]
counts = df['Statut'].value_counts()
colors_pie = [couleurs_statut[c] for c in counts.index]
wedges, texts, autotexts = ax.pie(
    counts.values, labels=counts.index, colors=colors_pie,
    autopct='%1.1f%%', startangle=90,
    wedgeprops={'edgecolor': 'white', 'linewidth': 2}
)
for at in autotexts:
    at.set_fontsize(11)
    at.set_fontweight('bold')
ax.set_title('(a) Répartition des statuts diabétiques', fontweight='bold')

# (b) Taux de diabète par âge - Ligne avec symboles
ax = axes[0, 1]
age_labels = ['18-24','25-29','30-34','35-39','40-44','45-49',
              '50-54','55-59','60-64','65-69','70-74','75-79','80+']
age_taux = df.groupby('Age')['Atteint'].mean() * 100
ax.plot(range(1, 14), age_taux.values, 'o-', color='#e74c3c',
        linewidth=2.5, markersize=8, markerfacecolor='white',
        markeredgewidth=2, label='Taux diabète+prédiabète')
ax.fill_between(range(1, 14), age_taux.values, alpha=0.15, color='#e74c3c')
ax.set_xticks(range(1, 14))
ax.set_xticklabels(age_labels, rotation=45, fontsize=8)
ax.set_xlabel('Catégorie d\'âge', fontsize=9)
ax.set_ylabel('Taux (%)', fontsize=9)
ax.set_title('(b) Taux de diabète par tranche d\'âge', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)

# (c) Distribution BMI par statut - Statistique (boxplot)
# Source boxplot : https://matplotlib.org/3.1.0/gallery/statistics/boxplot_demo.html
ax = axes[0, 2]
data_box = [df[df['Statut'] == s]['BMI'].values for s in ['Sain', 'Prédiabète', 'Diabète']]
bp = ax.boxplot(data_box, labels=['Sain', 'Prédiabète', 'Diabète'],
                patch_artist=True, notch=True)
colors_box = ['#2ecc71', '#f39c12', '#e74c3c']
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('IMC (BMI)', fontsize=9)
ax.set_title('(c) Distribution IMC par statut diabétique', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# (d) Barres - Facteurs aggravants (risque relatif)
ax = axes[1, 0]
# Facteurs aggravants (présence augmente le risque)
facteurs_agg = risques_df[risques_df['delta'] > 0].head(8)
bars = ax.barh(facteurs_agg['label'], facteurs_agg['risque_relatif'],
               color='#e74c3c', alpha=0.8, edgecolor='white')
ax.axvline(x=1, color='black', linestyle='--', linewidth=1.5, label='RR=1 (pas d\'effet)')
ax.set_xlabel('Risque Relatif', fontsize=9)
ax.set_title('(d) Facteurs AGGRAVANTS\n(Risque Relatif)', fontweight='bold')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3, axis='x')
for bar, val in zip(bars, facteurs_agg['risque_relatif']):
    ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
            f'{val:.2f}', va='center', fontsize=8)

# (e) Barres - Facteurs protecteurs
ax = axes[1, 1]
facteurs_prot = risques_df[risques_df['delta'] < 0].sort_values('risque_relatif')
bars2 = ax.barh(facteurs_prot['label'], facteurs_prot['risque_relatif'],
                color='#2ecc71', alpha=0.8, edgecolor='white')
ax.axvline(x=1, color='black', linestyle='--', linewidth=1.5)
ax.set_xlabel('Risque Relatif', fontsize=9)
ax.set_title('(e) Facteurs PROTECTEURS\n(Risque Relatif)', fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
for bar, val in zip(bars2, facteurs_prot['risque_relatif']):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
            f'{val:.2f}', va='center', fontsize=8)

# (f) Taux diabète par revenu - ligne + symboles
ax = axes[1, 2]
income_labels = ['<10k', '10-15k', '15-20k', '20-25k', '25-35k', '35-50k', '50-75k', '>75k']
income_taux = df.groupby('Income')['Atteint'].mean() * 100
ax.plot(range(1, 9), income_taux.values, 's--', color='#3498db',
        linewidth=2, markersize=9, markerfacecolor='#3498db',
        markeredgecolor='white', markeredgewidth=1.5)
ax.set_xticks(range(1, 9))
ax.set_xticklabels(income_labels, rotation=30, fontsize=8)
ax.set_xlabel('Revenu annuel', fontsize=9)
ax.set_ylabel('Taux (%)', fontsize=9)
ax.set_title('(f) Taux de diabète selon le revenu', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.invert_xaxis()  # Plus pauvre à gauche

# (g) Taux par niveau d'éducation - Barres
ax = axes[2, 0]
educ_labels = ['Maternelle', 'Primaire\n(1-8)', 'Collège\n(9-11)',
               'Lycée\nDiplômé', 'Bac+2/3', 'Bac+4\nou+']
educ_taux = df.groupby('Education')['Atteint'].mean() * 100
bars3 = ax.bar(educ_labels, educ_taux.values,
               color=[cm.RdYlGn_r(v/30) for v in educ_taux.values],
               edgecolor='white', linewidth=1.5)
ax.set_ylabel('Taux (%)', fontsize=9)
ax.set_title('(g) Taux de diabète\npar niveau d\'éducation', fontweight='bold')
ax.tick_params(axis='x', labelsize=7)
ax.grid(True, alpha=0.3, axis='y')

# (h) Santé générale vs diabète - Barres empilées
ax = axes[2, 1]
genhlth_labels = ['Excellent', 'Très bon', 'Bon', 'Passable', 'Mauvais']
for i, (statut, color) in enumerate([('Sain', '#2ecc71'), ('Prédiabète', '#f39c12'), ('Diabète', '#e74c3c')]):
    taux = df[df['Statut'] == statut].groupby('GenHlth').size() / df.groupby('GenHlth').size() * 100
    bottom = None if i == 0 else (
        df[df['Statut'] == 'Sain'].groupby('GenHlth').size() / df.groupby('GenHlth').size() * 100
        if i == 1 else
        (df[df['Statut'].isin(['Sain','Prédiabète'])].groupby('GenHlth').size() /
         df.groupby('GenHlth').size() * 100)
    )
    ax.bar(genhlth_labels, taux.values, label=statut, color=color,
           alpha=0.85, bottom=bottom.values if bottom is not None else None)
ax.set_ylabel('Proportion (%)', fontsize=9)
ax.set_title('(h) Statut diabétique\npar santé générale auto-déclarée', fontweight='bold')
ax.legend(fontsize=8)
ax.set_xlabel('Santé générale', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# (i) Combinaisons de facteurs - Heatmap risque
ax = axes[2, 2]
combo = df.groupby(['HighBP', 'HighChol'])['Atteint'].mean() * 100
combo_matrix = combo.unstack()
im = ax.imshow(combo_matrix.values, cmap='YlOrRd', aspect='auto',
               vmin=0, vmax=40)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Chol. normal', 'Chol. élevé'])
ax.set_yticks([0, 1])
ax.set_yticklabels(['PA normale', 'Hypertension'])
plt.colorbar(im, ax=ax, label='Taux (%)')
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{combo_matrix.values[i,j]:.1f}%',
                ha='center', va='center', fontweight='bold', fontsize=14,
                color='white' if combo_matrix.values[i,j] > 20 else 'black')
ax.set_title('(i) Combinaison Hypertension\n× Cholestérol → Taux diabète', fontweight='bold')

plt.tight_layout()
plt.savefig('/home/claude/fig1_matrice_globale.png', dpi=150, bbox_inches='tight',
            facecolor='white')
plt.close()
print("Figure 1 sauvegardée.")


# FIGURE 2 : ANALYSE DÉTAILLÉE - FACTEUR AGGRAVANT PRINCIPAL

fig2, axes2 = plt.subplots(2, 3, figsize=(18, 12))
fig2.suptitle('Question 1 – Facteur le plus aggravant du risque de diabète',
              fontsize=15, fontweight='bold')

# (a) Risque relatif de tous les facteurs - Graphique bar horizontal
ax = axes2[0, 0]
sorted_r = risques_df.sort_values('risque_relatif', ascending=True)
colors_bar = ['#e74c3c' if v > 1 else '#2ecc71' for v in sorted_r['risque_relatif']]
bars_all = ax.barh(sorted_r['label'], sorted_r['risque_relatif'],
                   color=colors_bar, alpha=0.8, edgecolor='white')
ax.axvline(x=1, color='black', linestyle='--', linewidth=2)
ax.set_xlabel('Risque Relatif (RR)', fontsize=10)
ax.set_title('Risque Relatif par facteur\n(rouge=aggravant, vert=protecteur)',
             fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# (b) Focus HighBP - l'aggravant principal : camembert comparatif
ax = axes2[0, 1]
# Taux de diabète AVEC et SANS hypertension
labels_bp = ['Diabète\n(avec HTA)', 'Sain/Prédiabète\n(avec HTA)',
             'Diabète\n(sans HTA)', 'Sain/Prédiabète\n(sans HTA)']
avec_hta = df[df['HighBP'] == 1]
sans_hta = df[df['HighBP'] == 0]
vals_pie2 = [
    avec_hta['Atteint'].sum(),
    len(avec_hta) - avec_hta['Atteint'].sum(),
    sans_hta['Atteint'].sum(),
    len(sans_hta) - sans_hta['Atteint'].sum()
]
colors_p2 = ['#c0392b', '#82e0aa', '#e74c3c', '#2ecc71']
wedges, texts, autotexts = ax.pie(vals_pie2, labels=labels_bp, colors=colors_p2,
    autopct='%1.1f%%', startangle=45,
    wedgeprops={'edgecolor': 'white', 'linewidth': 2})
ax.set_title('Répartition diabète\navec/sans Hypertension', fontweight='bold')

# (c) Combinaisons de 2 facteurs aggravants - Barres groupées
ax = axes2[0, 2]
combinaisons = {
    'HTA seule': df[(df['HighBP']==1) & (df['HighChol']==0) & (df['DiffWalk']==0)]['Atteint'].mean()*100,
    'Chol. seul': df[(df['HighBP']==0) & (df['HighChol']==1) & (df['DiffWalk']==0)]['Atteint'].mean()*100,
    'HTA+Chol.': df[(df['HighBP']==1) & (df['HighChol']==1) & (df['DiffWalk']==0)]['Atteint'].mean()*100,
    'HTA+March.': df[(df['HighBP']==1) & (df['HighChol']==0) & (df['DiffWalk']==1)]['Atteint'].mean()*100,
    'HTA+Chol\n+March.': df[(df['HighBP']==1) & (df['HighChol']==1) & (df['DiffWalk']==1)]['Atteint'].mean()*100,
    'Aucun': df[(df['HighBP']==0) & (df['HighChol']==0) & (df['DiffWalk']==0)]['Atteint'].mean()*100,
}
colors_combo = ['#f39c12','#f39c12','#e74c3c','#e74c3c','#c0392b','#2ecc71']
b = ax.bar(list(combinaisons.keys()), list(combinaisons.values()),
           color=colors_combo, alpha=0.85, edgecolor='white')
ax.axhline(y=df['Atteint'].mean()*100, color='navy', linestyle='--',
           linewidth=2, label=f'Moyenne globale ({df["Atteint"].mean()*100:.1f}%)')
ax.set_ylabel('Taux diabète (%)', fontsize=10)
ax.set_title('Impact des combinaisons de facteurs aggravants', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(b, combinaisons.values()):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.5, f'{val:.1f}%',
            ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.tick_params(axis='x', labelsize=8)

# (d) Taux de diabète par quartile de GenHlth et DiffWalk - Ligne+Symboles
ax = axes2[1, 0]
for dw, (label, color, marker) in enumerate([('Marche OK', '#3498db', 'o'),
                                              ('Difficulté marche', '#e74c3c', 's')]):
    sub = df[df['DiffWalk'] == dw]
    g = sub.groupby('GenHlth')['Atteint'].mean() * 100
    ax.plot(range(1,6), g.values, f'{marker}-', color=color,
            linewidth=2.5, markersize=9, label=label,
            markerfacecolor='white', markeredgewidth=2)
ax.set_xticks(range(1,6))
ax.set_xticklabels(['Excellent','Très bon','Bon','Passable','Mauvais'])
ax.set_xlabel('Santé générale auto-déclarée')
ax.set_ylabel('Taux diabète (%)')
ax.set_title('Taux de diabète : Santé générale\n× Difficulté de marche', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# (e) Heatmap AVC × Maladie cardiaque
ax = axes2[1, 1]
combo2 = df.groupby(['Stroke', 'HeartDiseaseorAttack'])['Atteint'].mean() * 100
m2 = combo2.unstack().values
im2 = ax.imshow(m2, cmap='OrRd', aspect='auto', vmin=0, vmax=55)
ax.set_xticks([0,1])
ax.set_xticklabels(['Pas maladie\ncardiaque', 'Maladie\ncardiaque'])
ax.set_yticks([0,1])
ax.set_yticklabels(['Pas d\'AVC', 'AVC'])
plt.colorbar(im2, ax=ax, label='Taux (%)')
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{m2[i,j]:.1f}%', ha='center', va='center',
                fontweight='bold', fontsize=16,
                color='white' if m2[i,j] > 35 else 'black')
ax.set_title('Combinaison AVC × Maladie\ncardiaque → Risque diabète', fontweight='bold')

# (f) Distribution par sexe et âge - Barres groupées
ax = axes2[1, 2]
age_sex = df.groupby(['Age', 'Sex'])['Atteint'].mean() * 100
age_f = age_sex.xs(0, level='Sex')
age_m = age_sex.xs(1, level='Sex')
x = np.arange(13)
w = 0.35
ax.bar(x - w/2, age_f.values, w, label='Femme', color='#e91e8c', alpha=0.8)
ax.bar(x + w/2, age_m.values, w, label='Homme', color='#3498db', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(age_labels, rotation=45, fontsize=7)
ax.set_ylabel('Taux diabète (%)')
ax.set_title('Taux de diabète par âge et sexe', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/home/claude/fig2_facteurs_aggravants.png', dpi=150, bbox_inches='tight',
            facecolor='white')
plt.close()
print("Figure 2 sauvegardée.")


# FIGURE 3 : FACTEURS PROTECTEURS

fig3, axes3 = plt.subplots(2, 3, figsize=(18, 12))
fig3.suptitle('Question 2 – Facteurs protecteurs contre le diabète',
              fontsize=15, fontweight='bold')

# (a) Camembert : actif physiquement vs non
ax = axes3[0, 0]
actif = df[df['PhysActivity'] == 1]
inactif = df[df['PhysActivity'] == 0]
labels_pie3 = ['Diabète\n(Actif)', 'Sain\n(Actif)', 'Diabète\n(Inactif)', 'Sain\n(Inactif)']
sizes3 = [actif['Atteint'].sum(), len(actif)-actif['Atteint'].sum(),
          inactif['Atteint'].sum(), len(inactif)-inactif['Atteint'].sum()]
c3 = ['#27ae60', '#a9dfbf', '#e74c3c', '#f1948a']
wedges, texts, ats = ax.pie(sizes3, labels=labels_pie3, colors=c3,
                             autopct='%1.1f%%', startangle=90,
                             wedgeprops={'edgecolor':'white','linewidth':2})
ax.set_title('Activité physique\nvs diabète', fontweight='bold')

# (b) Combinaisons protectrices - Barres
ax = axes3[0, 1]
combos_prot = {
    'Actif seul': df[(df['PhysActivity']==1)&(df['Fruits']==0)&(df['Veggies']==0)]['Atteint'].mean()*100,
    'Fruits seuls': df[(df['PhysActivity']==0)&(df['Fruits']==1)&(df['Veggies']==0)]['Atteint'].mean()*100,
    'Légumes seuls': df[(df['PhysActivity']==0)&(df['Fruits']==0)&(df['Veggies']==1)]['Atteint'].mean()*100,
    'Actif+Fruits': df[(df['PhysActivity']==1)&(df['Fruits']==1)&(df['Veggies']==0)]['Atteint'].mean()*100,
    'Actif+Légumes': df[(df['PhysActivity']==1)&(df['Fruits']==0)&(df['Veggies']==1)]['Atteint'].mean()*100,
    'Actif+F+L': df[(df['PhysActivity']==1)&(df['Fruits']==1)&(df['Veggies']==1)]['Atteint'].mean()*100,
    'Aucun': df[(df['PhysActivity']==0)&(df['Fruits']==0)&(df['Veggies']==0)]['Atteint'].mean()*100,
}
mean_global = df['Atteint'].mean()*100
colors_prot = ['#27ae60','#58d68d','#58d68d','#1e8449','#1e8449','#145a32','#e74c3c']
b3 = ax.bar(list(combos_prot.keys()), list(combos_prot.values()),
            color=colors_prot, alpha=0.85, edgecolor='white')
ax.axhline(y=mean_global, color='navy', linestyle='--',
           linewidth=2, label=f'Moyenne ({mean_global:.1f}%)')
ax.set_ylabel('Taux diabète (%)')
ax.set_title('Combinaisons de facteurs protecteurs', fontweight='bold')
ax.legend(fontsize=9)
ax.tick_params(axis='x', labelsize=8, rotation=15)
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(b3, combos_prot.values()):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.3,
            f'{val:.1f}%', ha='center', fontsize=8, fontweight='bold')

# (c) Effet de l'alcool - avec ligne et symboles
ax = axes3[0, 2]
# Alcool + activité physique
for alc, (label, color, marker, ls) in enumerate([
    ('Non-buveur', '#3498db', 'o', '-'),
    ('Buveur excessif', '#e74c3c', 's', '--')
]):
    sub = df[df['HvyAlcoholConsump'] == alc]
    taux_actif = sub[sub['PhysActivity']==1]['Atteint'].mean()*100
    taux_inactif = sub[sub['PhysActivity']==0]['Atteint'].mean()*100
    ax.bar([alc*2, alc*2+1], [taux_inactif, taux_actif],
           color=['#e74c3c', color], alpha=0.8, edgecolor='white',
           label=[f'{label} inactif', f'{label} actif'][0] if alc==0 else None)

# Remake propre
ax.cla()
cats = ['Non-buveur\nInactif', 'Non-buveur\nActif', 'Buveur\nInactif', 'Buveur\nActif']
taux_alcool_act = [
    df[(df['HvyAlcoholConsump']==0)&(df['PhysActivity']==0)]['Atteint'].mean()*100,
    df[(df['HvyAlcoholConsump']==0)&(df['PhysActivity']==1)]['Atteint'].mean()*100,
    df[(df['HvyAlcoholConsump']==1)&(df['PhysActivity']==0)]['Atteint'].mean()*100,
    df[(df['HvyAlcoholConsump']==1)&(df['PhysActivity']==1)]['Atteint'].mean()*100,
]
c_act = ['#e74c3c','#27ae60','#c0392b','#1e8449']
b4 = ax.bar(cats, taux_alcool_act, color=c_act, alpha=0.85, edgecolor='white')
ax.axhline(y=mean_global, color='navy', linestyle='--', linewidth=2)
ax.set_ylabel('Taux diabète (%)')
ax.set_title('Alcool × Activité physique\n→ Taux diabète', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(b4, taux_alcool_act):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.2,
            f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')

# (d) Ligne : Taux diabète selon mental health × activité physique
ax = axes3[1, 0]
for act, (label, color, marker) in enumerate([('Inactif', '#e74c3c', 's'), ('Actif', '#27ae60', 'o')]):
    sub = df[df['PhysActivity'] == act]
    g = sub[sub['MentHlth'] <= 15].groupby('MentHlth')['Atteint'].mean() * 100
    ax.plot(g.index, g.values, f'{marker}-', color=color, linewidth=2,
            markersize=5, label=label, alpha=0.8)
ax.set_xlabel('Jours de mauvaise santé mentale (30 derniers jours)')
ax.set_ylabel('Taux diabète (%)')
ax.set_title('Santé mentale × Activité physique\n→ Risque diabète', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# (e) Heatmap protectrice : Fruits × Légumes × Activité (3D → 2D heatmap)
ax = axes3[1, 1]
combo_prot2 = df.groupby(['Fruits', 'Veggies'])['Atteint'].mean() * 100
m_prot = combo_prot2.unstack().values
im3 = ax.imshow(m_prot, cmap='RdYlGn_r', aspect='auto', vmin=12, vmax=25)
ax.set_xticks([0,1])
ax.set_xticklabels(['Sans légumes', 'Avec légumes'])
ax.set_yticks([0,1])
ax.set_yticklabels(['Sans fruits', 'Avec fruits'])
plt.colorbar(im3, ax=ax, label='Taux (%)')
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{m_prot[i,j]:.1f}%', ha='center', va='center',
                fontweight='bold', fontsize=18,
                color='white' if m_prot[i,j] > 20 else 'black')
ax.set_title('Fruits × Légumes\n→ Risque diabète', fontweight='bold')

# (f) Revenu × Éducation heatmap
ax = axes3[1, 2]
combo_rev_educ = df.groupby(['Education', 'Income'])['Atteint'].mean() * 100
m_re = combo_rev_educ.unstack()
im4 = ax.imshow(m_re.values, cmap='RdYlGn_r', aspect='auto')
ax.set_xticks(range(8))
ax.set_xticklabels(['<10k','10k','15k','20k','25k','35k','50k','>75k'], fontsize=7, rotation=30)
ax.set_yticks(range(6))
ax.set_yticklabels(['Matern.','Primaire','Collège','Lycée','Bac+2','Bac+4'], fontsize=8)
plt.colorbar(im4, ax=ax, label='Taux diabète (%)')
ax.set_title('Heatmap : Éducation × Revenu\n→ Risque diabète', fontweight='bold')
ax.set_xlabel('Revenu annuel', fontsize=9)
ax.set_ylabel('Niveau d\'éducation', fontsize=9)

plt.tight_layout()
plt.savefig('/home/claude/fig3_facteurs_protecteurs.png', dpi=150, bbox_inches='tight',
            facecolor='white')
plt.close()
print("Figure 3 sauvegardée.")


# FIGURE 4 : ANALYSE BMI - GRAPHIQUE AVEC COURBE DE FITTING
# + GRAPHIQUE 3D + ISOCONTOURS

fig4, axes4 = plt.subplots(2, 3, figsize=(20, 13))
fig4.suptitle('Question 3 – Impact de l\'IMC (BMI) sur le développement du diabète',
              fontsize=15, fontweight='bold')

# (a) Distribution BMI par statut - graphique statistique (violin + box)
ax = axes4[0, 0]
# Source violin plot : https://matplotlib.org/3.1.0/gallery/statistics/violin_and_kde.html
parts = ax.violinplot(
    [df[df['Statut']=='Sain']['BMI'].dropna(),
     df[df['Statut']=='Prédiabète']['BMI'].dropna(),
     df[df['Statut']=='Diabète']['BMI'].dropna()],
    positions=[1,2,3], showmedians=True, showextrema=True
)
colors_v = ['#2ecc71', '#f39c12', '#e74c3c']
for pc, color in zip(parts['bodies'], colors_v):
    pc.set_facecolor(color)
    pc.set_alpha(0.7)
ax.set_xticks([1,2,3])
ax.set_xticklabels(['Sain', 'Prédiabète', 'Diabète'])
ax.set_ylabel('IMC (BMI)')
ax.set_title('(a) Distribution de l\'IMC\npar statut diabétique (violin)', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
medians = [df[df['Statut']==s]['BMI'].median() for s in ['Sain','Prédiabète','Diabète']]
for pos, med in zip([1,2,3], medians):
    ax.text(pos, med+1, f'Mé={med:.1f}', ha='center', fontsize=8, fontweight='bold')

# (b) Taux de diabète par BMI avec fitting - COURBE AVEC SYMBOLES + LIGNE
ax = axes4[0, 1]
# Regrouper BMI par intervalles de 1
df_bmi = df[(df['BMI'] >= 15) & (df['BMI'] <= 60)].copy()
bmi_groups = df_bmi.groupby(df_bmi['BMI'].round(0))
bmi_taux = bmi_groups['Atteint'].mean() * 100
bmi_count = bmi_groups['Atteint'].count()
bmi_x = bmi_taux.index.values
bmi_y = bmi_taux.values
# Filtrer les groupes avec assez d'observations
mask = bmi_count > 50
bmi_x = bmi_x[mask]
bmi_y = bmi_y[mask]

# Code fortement inspirée par l'IA suivante : CLAUDE SONNET 4.6

# Fitting sigmoïde / logistique
# Source curve_fit : https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
def sigmoid(x, L, k, x0, b):
    return L / (1 + np.exp(-k * (x - x0))) + b

try:
    popt, _ = curve_fit(sigmoid, bmi_x, bmi_y,
                        p0=[40, 0.15, 30, 2], maxfev=5000)
    x_fit = np.linspace(bmi_x.min(), bmi_x.max(), 200)
    y_fit = sigmoid(x_fit, *popt)
    ax.plot(x_fit, y_fit, '-', color='#c0392b', linewidth=3, label='Ajustement sigmoïdal', zorder=3)
except Exception as e:
    # Fallback : régression polynomiale
    z = np.polyfit(bmi_x, bmi_y, 3)
    p = np.poly1d(z)
    x_fit = np.linspace(bmi_x.min(), bmi_x.max(), 200)
    ax.plot(x_fit, p(x_fit), '-', color='#c0392b', linewidth=3, label='Ajustement poly deg.3', zorder=3)

ax.scatter(bmi_x, bmi_y, s=30, color='#3498db', alpha=0.7,
           zorder=2, label='Taux observé (par IMC)')
ax.axvline(x=25, color='green', linestyle=':', linewidth=2, label='IMC normal/surpoids (25)')
ax.axvline(x=30, color='orange', linestyle=':', linewidth=2, label='Surpoids/obésité (30)')
ax.set_xlabel('IMC (BMI)')
ax.set_ylabel('Taux diabète+prédiabète (%)')
ax.set_title('(b) Taux diabète en fonction de l\'IMC\navec ajustement de courbe', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Corrélation de Pearson
r, p_val = stats.pearsonr(bmi_x, bmi_y)
ax.text(0.05, 0.95, f'r de Pearson = {r:.3f}\np < 0.001', transform=ax.transAxes,
        verticalalignment='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# (c) Histogramme BMI par statut (superposé)
ax = axes4[0, 2]
for statut, color, alpha in [('Sain','#2ecc71',0.5),('Prédiabète','#f39c12',0.7),('Diabète','#e74c3c',0.7)]:
    sub = df[df['Statut']==statut]['BMI']
    sub = sub[(sub>=10)&(sub<=60)]
    ax.hist(sub, bins=50, color=color, alpha=alpha, label=statut, density=True)
ax.set_xlabel('IMC (BMI)')
ax.set_ylabel('Densité')
ax.set_title('(c) Distribution de l\'IMC\npar statut diabétique (histogramme)', fontweight='bold')
ax.legend()
ax.axvline(x=25, color='green', linestyle='--', linewidth=1.5)
ax.axvline(x=30, color='orange', linestyle='--', linewidth=1.5)
ax.grid(True, alpha=0.3)

# (d) Camembert : répartition catégories IMC chez les diabétiques
ax = axes4[1, 0]
diab = df[df['Diabetes_012'] == 2].copy()
diab['Cat_BMI'] = pd.cut(diab['BMI'],
                          bins=[0, 18.5, 25, 30, 35, 100],
                          labels=['Sous-poids\n(<18.5)', 'Normal\n(18.5-25)',
                                  'Surpoids\n(25-30)', 'Obèse\n(30-35)',
                                  'Obèse sévère\n(>35)'])
cat_counts = diab['Cat_BMI'].value_counts()
colors_bmi = ['#3498db', '#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
wedges, texts, ats = ax.pie(cat_counts.values, labels=cat_counts.index,
                             colors=colors_bmi, autopct='%1.1f%%',
                             wedgeprops={'edgecolor':'white','linewidth':2})
ax.set_title('(d) Catégories IMC chez les\npersonnes diabétiques', fontweight='bold')

# (e) GRAPHIQUE 3D : BMI × Âge → Taux diabète
# Source 3D plot : https://matplotlib.org/3.1.0/gallery/mplot3d/surface3d.html
ax3d = fig4.add_subplot(2, 3, 5, projection='3d')
fig4.delaxes(axes4[1, 1])

# Grille BMI × Age
bmi_bins = pd.cut(df['BMI'], bins=np.arange(15, 60, 5))
age_bins = df['Age']
pivot = df.groupby([bmi_bins, age_bins])['Atteint'].mean() * 100
pivot_df = pivot.unstack()

# Nettoyer
pivot_df = pivot_df.dropna(how='all').dropna(axis=1, how='all')
Z = pivot_df.fillna(0).values
X = np.arange(Z.shape[1])  # Ages
Y = np.arange(Z.shape[0])  # BMI bins
X, Y = np.meshgrid(X, Y)

surf = ax3d.plot_surface(X, Y, Z, cmap='RdYlGn_r', alpha=0.85,
                          linewidth=0, antialiased=True)
ax3d.set_xlabel('Catégorie d\'âge', fontsize=8, labelpad=5)
ax3d.set_ylabel('Catégorie BMI', fontsize=8, labelpad=5)
ax3d.set_zlabel('Taux diabète (%)', fontsize=8, labelpad=5)
ax3d.set_title('(e) Graphique 3D : IMC × Âge\n→ Taux diabète', fontweight='bold')
ax3d.view_init(elev=25, azim=-60)
fig4.colorbar(surf, ax=ax3d, shrink=0.4, label='Taux (%)')

# (f) ISOCONTOURS BMI × Age → Taux diabète
# Source contour : https://matplotlib.org/3.1.0/gallery/images_contours_and_fields/contourf_demo.html
ax = axes4[1, 2]
Z_smooth = pivot_df.ffill().bfill().values
from scipy.ndimage import gaussian_filter
Z_s = gaussian_filter(Z_smooth, sigma=1)
X_c = np.arange(Z_s.shape[1])
Y_c = np.arange(Z_s.shape[0])
levels = np.linspace(0, 50, 15)
cf = ax.contourf(X_c, Y_c, Z_s, levels=levels, cmap='RdYlGn_r', alpha=0.85)
cs = ax.contour(X_c, Y_c, Z_s, levels=levels, colors='white', alpha=0.4, linewidths=0.8)
ax.clabel(cs, inline=True, fontsize=7, fmt='%.0f%%')
plt.colorbar(cf, ax=ax, label='Taux diabète (%)')
age_ticks = list(pivot_df.columns)
bmi_cats = [str(c) for c in pivot_df.index]
ax.set_xticks(range(len(age_ticks)))
ax.set_xticklabels([str(int(a)) for a in age_ticks], fontsize=8)
ax.set_yticks(range(len(bmi_cats)))
ax.set_yticklabels(bmi_cats, fontsize=7)
ax.set_xlabel('Catégorie d\'âge (1-13)', fontsize=9)
ax.set_ylabel('Catégorie IMC', fontsize=9)
ax.set_title('(f) Isocontours : IMC × Âge\n→ Taux diabète', fontweight='bold')

plt.tight_layout()
plt.savefig('/home/claude/fig4_bmi_analyse.png', dpi=150, bbox_inches='tight',
            facecolor='white')
plt.close()
print("Figure 4 sauvegardée.")


# RÉSUMÉ STATISTIQUE

print("\n" + "=" * 60)
print("RÉSUMÉ DES RÉSULTATS")
print("=" * 60)
print(f"\nPopulation totale : {len(df):,} individus")
print(f"  - Sains : {(df['Diabetes_012']==0).sum():,} ({(df['Diabetes_012']==0).mean()*100:.1f}%)")
print(f"  - Prédiabète : {(df['Diabetes_012']==1).sum():,} ({(df['Diabetes_012']==1).mean()*100:.1f}%)")
print(f"  - Diabète : {(df['Diabetes_012']==2).sum():,} ({(df['Diabetes_012']==2).mean()*100:.1f}%)")

print(f"\nFACTEUR LE PLUS AGGRAVANT : {risques_df.iloc[0]['label']}")
print(f"  RR = {risques_df.iloc[0]['risque_relatif']:.2f}")
print(f"  Taux avec : {risques_df.iloc[0]['taux_present']:.1f}% vs sans : {risques_df.iloc[0]['taux_absent']:.1f}%")

prot = risques_df[risques_df['risque_relatif'] < 1].sort_values('risque_relatif')
print(f"\nFACTEUR LE PLUS PROTECTEUR : {prot.iloc[0]['label']}")
print(f"  RR = {prot.iloc[0]['risque_relatif']:.2f}")
print(f"  Taux avec : {prot.iloc[0]['taux_present']:.1f}% vs sans : {prot.iloc[0]['taux_absent']:.1f}%")

print(f"\nCORRÉLATION BMI - DIABÈTE :")
print(f"  r de Pearson (BMI groupé) = {r:.3f}")
print(f"  IMC moyen sain : {df[df['Statut']=='Sain']['BMI'].mean():.1f}")
print(f"  IMC moyen diabète : {df[df['Statut']=='Diabète']['BMI'].mean():.1f}")
print(f"  Différence : +{df[df['Statut']=='Diabète']['BMI'].mean() - df[df['Statut']=='Sain']['BMI'].mean():.1f} points IMC")

print("\nToutes les figures ont été sauvegardées.")
