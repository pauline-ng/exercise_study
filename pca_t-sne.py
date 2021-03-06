import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import seaborn as sns
from sklearn.preprocessing import StandardScaler

import os

def add_age_gender (df, xls_file):
    xl = pd.ExcelFile(xls_file)
    df_body = xl.parse("Body measurements")
    df_body = df_body.rename(columns={'Subject': 'SUBJECT'})

    df_day0 = df_body.pivot('SUBJECT', 'Measurement parameter', 'Value: 09 Sep')
#    print (df_day0.head())
    df_day0.reset_index()
    df_day0.columns.name=None
    df_day0.astype ('float').dtypes
    df = df.merge (df_day0,  on="SUBJECT", how='left')

    df_age_gender = df_body[['SUBJECT', 'Age', 'Gender']].drop_duplicates()
    df_age_gender['Age_group'] = df_age_gender.apply(lambda row: 'Under50' if row['Age'] < 50 else 'Over50', axis=1)
    df_age_gender['Age_Gender'] = df_age_gender.apply (lambda row: ' '.join ([row['Gender'], row['Age_group']]),axis=1)
    df_age_gender['Gender_numerical'] =  df_age_gender.apply (lambda row: 1 if row['Gender'] == 'M'else 0, axis=1)
    print(df_age_gender.head())
    df = df.merge(df_age_gender, on="SUBJECT", how='left')
   # print (df.columns)
    return df

def read_metabolite_data (xls_file):
    outdir = "/home/pauline/exercise_study/"

    xl = pd.ExcelFile(xls_file)


    df_clinical_lipids = xl.parse("Clinical lipids")

    # add CRP sheet
    df_crp = xl.parse("CRP")
    df_clinical_lipids = df_clinical_lipids.append(df_crp)
    print ("CRP")
    print(df_crp['DATE'].value_counts())

    # add TSH sheet
    df_TSH = xl.parse("TSH")
    df_clinical_lipids = df_clinical_lipids.append(df_TSH[['SUBJECT', 'DATE', 'TIME', 'TEST', 'RES', 'UNIT']])
    print ("TSH")
    print(df_TSH['DATE'].value_counts())

    # add FBC sheet
    df_FBC = xl.parse("FBC")
    # some cleanup on dates
 #   df_FBC.iloc[df_FBC['DATE' == "07/09/2020"]]['DATE'] = "09/09/2020"
 #   df_FBC.iloc[df_FBC['DATE' == "03/11/2020"]]['DATE'] = "02/11/2020"
    df_clinical_lipids = df_clinical_lipids.append(df_FBC[['SUBJECT', 'DATE', 'TIME', 'TEST', 'RES', 'UNIT']])
    df_clinical_lipids.reset_index(drop=True)
    print ("FBC")
    print(df_FBC['DATE'].value_counts())
    # drop rows where there's no data
    df_clinical_lipids = df_clinical_lipids[df_clinical_lipids['RES'].notna()]

    # some cleanup on dates for people that were a day off
    # the first line is a must because FBC was measured twice, first on Sept 7 and on Sept 9
    # but all the other measurements TSH and CRP were one Sept 9
    df_clinical_lipids.loc[df_clinical_lipids['DATE'] == "07/09/2020",'DATE'] = "09/09/2020"
    df_clinical_lipids.loc[df_clinical_lipids['DATE'] == "03/11/2020",'DATE'] = "02/11/2020"
    df_clinical_lipids.loc[df_clinical_lipids['DATE'] == "07/12/2020",'DATE'] = "14/12/2020" # ask Anne about this one, it's more than 1 day off
    df_clinical_lipids.loc[df_clinical_lipids['DATE'] == "20/10/2020",'DATE'] = "19/10/2020"
    df_clinical_lipids.loc[df_clinical_lipids['DATE'] == "13/10/2020",'DATE'] = "12/10/2020"
    df_clinical_lipids.loc[df_clinical_lipids['DATE'] == "24/11/2020", 'DATE'] = "23/11/2020"
    print (df_clinical_lipids['DATE'].value_counts())

    # turn Date into integer to allow for linear regression
    df_clinical_lipids["Day"] = df_clinical_lipids.apply(
        lambda row: datetime.strptime(row["DATE"], '%d/%m/%Y').timetuple().tm_yday, axis=1)
    df_clinical_lipids["Day"] = df_clinical_lipids["Day"].astype(int)

    # clean out non-numeric entries
    df_clinical_lipids = df_clinical_lipids[
        df_clinical_lipids['RES'].apply(lambda x: type(x) in [int, np.int64, float, np.float64])]

    # get first day, then normalize to that
    first_day = df_clinical_lipids["Day"].min()
    df_clinical_lipids["Day"] = df_clinical_lipids["Day"] - first_day
    grouped = df_clinical_lipids.groupby(["TEST"])
    dfs = [x for _, x in df_clinical_lipids.groupby('TEST')]
    avg_dfs_list = [] # averaging measurement for each day
    for df in dfs:
        test = df["TEST"].iloc[0]
        print(test)

        # now convert it back to float
        df['RES'] = df['RES'].astype(float)


        avg_df = pd.DataFrame({'RES': df.groupby(['SUBJECT', 'Day'])['RES'].mean()}).reset_index()
        avg_df['TEST'] = test
        avg_dfs_list.append (avg_df)
    combined_df = pd.concat (avg_dfs_list)
    return combined_df

def run_PCA (df, scaler, pca, feat_cols, pca_components, fit_transform = True):
    print ('running PCA')

    #autoscaler = StandardScaler()
    #autoscaled_df = autoscaler.fit_transform (df[feat_cols])
    autoscaled_df = scaler.transform (df[feat_cols])
    #print ("autoscale")
    #print (autoscaled_df)
    num_components = pca_components

   # print (df[feat_cols].values)
   # pca_result = pca.fit_transform (df[feat_cols].values)
    if fit_transform:
        pca_result = pca.fit_transform(autoscaled_df)
        # scree plot
        var_df = pd.DataFrame({'var': pca.explained_variance_ratio_,
                           'PC': ["PC{}".format (x) for x in range (1,pca_num_components+1)] })
        ax = sns.barplot(x='PC', y="var",
                    data=var_df, color="c")
        fig = ax.get_figure()
        fig.savefig ('PCA_scree.png')
        fig.clf()
    else:
        pca_result = pca.transform (autoscaled_df)
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

    # add PCA to sample
    for i in range (0, num_components):
        df['PCA' + str(i+1)] = pca_result[:,i]
    return pca_result

def plot_TSNE (df, tsne_results, outfig):
    df['TSNE1'] = tsne_results[:,0]
    df['TSNE2'] = tsne_results[:,1]
    plt.figure (figsize=(16,10))
    sns.scatterplot (x="TSNE1", y="TSNE2",
    #hue="SUBJECT", \
    palette=sns.color_palette('hls', len(df.index)), data=df, legend="full", alpha=0.3)
    plt.savefig (outfig)
    plt.clf()

def plot_PCA (df, outfig_base):
    plt.figure (figsize=(16,10))
#    print (df['SUBJECT'].nunique())
    print (df.head())
    color_dict = dict({'F Over50': 'orange', 'F Under50': 'green', 'M Over50': 'brown', 'M Under50': 'black'})
    sns.scatterplot (x="PCA1", y="PCA2", hue="Age_Gender", \
#    palette=['green', 'orange', 'brown', 'red'], \
                    palette = color_dict,\
                     data=df, legend="full") #alpha = 0.3 (transparency)
    plt.savefig (outfig_base + ".PCA_2D.png")
    plt.clf()

    #3D

    ax = plt.figure (figsize=(16,10)).gca (projection='3d')
    #print (list(df['Age_Gender']))
    colors=[]
    for i in list (df['Age_Gender']):
        if i in color_dict:
            colors.append (color_dict[i])
        else:
            colors.append ("white")
    #print (colors)
    ax.scatter (xs =df["PCA1"], ys=df["PCA2"], zs=df["PCA3"] , c=colors) #, cmap=color_dict)
    ax.set_xlabel ('PC1')
    ax.set_ylabel ('PC2')
    ax.set_zlabel ('PC3')
    #, data=df)
    #hue="SUBJECT", \
    #palette=sns.color_palette('hls', len(df.index)), data=df, legend="full", alpha=0.3)
    plt.savefig (outfig_base + ".PCA_3D.png")
    plt.clf()

def plot_trajectories (df_list, outfig):
    fig = plt.figure(figsize=(16,20))
#    ax_2D = fig.add_subplot (1,2,1)
    ax_3D = fig.add_subplot(3,1, (1,2), projection='3d')
    ax_2D = fig.add_subplot(3, 1, 3)
    fig_2D = plt.figure()
    fig_3D = plt.figure(figsize=(16,10)).gca (projection='3d')

   # ax_2D = fig_2D.add_subplot(111)
   # ax_3D = fig_3D #.add_subplot(111)

    first_df = df_list[0]
   # print (first_df.columns)
    pc_df = first_df[['SUBJECT','PCA1', 'PCA2','PCA3']]
    for df_index in range (1, len(df_list)):
        df = df_list[df_index]
        pc_df = pc_df.merge (df[['SUBJECT', 'PCA1', 'PCA2', 'PCA3']], on="SUBJECT", how='left')
 #   ax = plt.figure (figsize=(16,10)).gca (projection='3d')

    pca1_columns = pc_df.columns[pc_df.columns.str.contains ("PCA1")]
    pca2_columns = [col for col in pc_df.columns if col.startswith ("PCA2")]
    pca3_columns = [col for col in pc_df.columns if col.startswith("PCA3")]

    PC1_array = pc_df[pca1_columns].to_numpy()
    PC2_array = pc_df[pca2_columns].to_numpy()
    PC3_array = pc_df[pca3_columns].to_numpy()
    #print (pca1_columns)
    colors = sns.color_palette('muted', n_colors=len(PC1_array))

    for sample_index in range (0, len(PC1_array)):

        sample_color = colors[sample_index]
        rgb = [ int (element * 255) for element in list (sample_color)]
        #print (rgb)
        hex_color =  '#%02x%02x%02x' % tuple (rgb)
        #print (hex_color )
        #print ("PC1")
        #print (PC1_array[sample_index])
        #print ("PC2")
        #print (PC2_array[sample_index])
        #print ("PC3")
        #print (PC3_array[sample_index])
        ax_2D.plot (PC1_array[sample_index],PC2_array[sample_index], color = hex_color)
        ax_3D.plot (PC1_array[sample_index],PC2_array[sample_index],PC3_array[sample_index], \
                 color=hex_color) #c= sample_color ) # need to assign color c=colors
        #first point
        ax_3D.scatter(PC1_array[sample_index][0], PC2_array[sample_index][0], PC3_array[sample_index][0], marker="o", c=hex_color) #sample_color)

        #
       #last point
        ax_3D.scatter (PC1_array[sample_index][-1],PC2_array[sample_index][-1],PC3_array[sample_index][-1],marker=">", c=hex_color) #sample_color )
        ax_2D.scatter (PC1_array[sample_index][-1],PC2_array[sample_index][-1], marker=">", c=hex_color)
    ax_3D.set_xlabel ('PC1')
    ax_3D.set_ylabel ('PC2')
    ax_3D.set_zlabel ('PC3')
    ax_2D.set_xlabel ('PC1')
    ax_2D.set_ylabel ('PC2')
    #, data=df)
    #hue="SUBJECT", \
    #palette=sns.color_palette('hls', len(df.index)), data=df, legend="full", alpha=0.3)
    fig.savefig (outfig)
    print ("figure saved in " + outfig)
    plt.clf()

def run_TSNE (df, pca_result):
    tsne = TSNE (n_components=2, verbose=0, perplexity=40, n_iter=300)
    tsne_pca_results = tsne.fit_transform (pca_result)
    return tsne_pca_results

def print_PC_correlates (df, pca_num_components, pca_results_file):
    corr = df.corr()
    corr_abs = corr.abs()
 #   s = corr_abs.unstack ()
    s = corr.unstack()
    so = s.to_frame().reset_index()
    print ("PC correlates")
   # so = s.sort_values (kind="quicksort", ascending=False)
    print (type (so))
    #so = so.to_frame().reset_index()
    so.columns = ['x','y','corr']
    so['abs(corr)'] = so.apply (lambda row: abs (row['corr']),axis=1)
    print(so.head())
   # exit()
#    print (so[0])
    so = so[so['x'] != so['y']]
    so = so[so['abs(corr)'] >= 0.5]
    so = so[so['x'].str.contains ("PC")]
    so= so.sort_values (by=['x', 'abs(corr)'], ascending=[True, False])
    print (so.head())
    so.to_csv (pca_results_file, sep="\t", index=False, float_format="%.3f")

def pivot_on_test_res (df):
    df_pivot = df.pivot('SUBJECT', 'TEST', 'RES')
    df_pivot.reset_index()
    df_pivot.columns.name = None
    return df_pivot

def apply_pca (df_day, scaler, pca, pca_num_components, feat_cols, xls_file):
    df_day_pivot = pivot_on_test_res (df_day)
    df_day_pivot = add_age_gender(df_day_pivot, xls_file)
    #this is bad coding, age & gender are port of feat_cols. should probably be added  before apply_pca

    df_day_pivot.dropna (subset=feat_cols, inplace=True)
    #print ("drop na")
    #print (df_day_pivot.shape)
    pca_result = run_PCA (df_day_pivot, scaler, pca, feat_cols, pca_num_components, False)
    print ("after PCA")
    #print (df_day_pivot.shape)
    #df_day_pivot = add_age_gender(df_day_pivot,xls_file)
    return df_day_pivot

def plot_loadings (score, coeff, labels, outfig):
    xs = score[:, 0]
    ys = score[:, 1]
    n = coeff.shape[0]

    #plt.scatter(xs, ys, c=y)  # without scaling
    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5)
        if labels is None:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, "Var" + str(i + 1), color='g', ha='center', va='center')
        else:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, labels[i], color='g', ha='center', va='center')

    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
   # plt.grid()
    plt.savefig (outfig)
    plt.clf()

def pca_on_time_points (list_of_dfs)
    df_day_0 = list_of_dfs[0]

if __name__ == '__main__':
    xls_file = "/home/pauline/exercise_study_data/ExerciseStudy.xls"
    outdir = "/home/pauline/exercise_study/"
    df = read_metabolite_data(xls_file)
    df_day0 = df[df['Day'] == 0]
    df_day96 = df[df['Day'] == 96]
    df_day54 = df[df['Day'] == 54]
    #print (df['Day'].value_counts())
 #   print (df_day54['TEST'].value_counts())
    #exit()_
 #   print (df_day0.head())
  #  df_day0_pivot = df_day0.pivot ('SUBJECT', 'TEST', 'RES')
   # df_day0_pivot['RES'].reset_index()
  #  df_day0_pivot.reset_index()
   # df_day0_pivot.columns.name=None
    df_day0_pivot = pivot_on_test_res(df_day0)
    df_day0_pivot = df_day0_pivot.dropna()  # drop rows with NaN

  #  df_day0_pivot.columns = df_day0_pivot.columns.droplevel().rename(None)
    print (df_day0_pivot.head())
    feat_cols = df_day0_pivot.columns.to_list()

    # TODO, comment below out once get new dataset. GHB and GHB2 missing for day 54
    feat_cols = [col  for col in feat_cols if col != 'GHB' and col != 'GHB2' and col != 'NRBC' and col != 'MPV' ]
  #  print ("hi there")
    print (feat_cols)
    # TODO might want to remove feat_cols correlated > 0.7?
    print ("before")
    print (df_day0_pivot.head())
    df_day0_pivot = add_age_gender(df_day0_pivot, xls_file)
    feat_cols.extend (['Age', 'Gender_numerical'])
    print (feat_cols)
    pca_num_components = 8
    pca = PCA(n_components=pca_num_components)
   # autoscaler = StandardScaler()
   # df_day0_pivot_all = df_day0_pivot
#    df_day0_pivot = df_day0_pivot[feat_cols] # clean it up
    # scale to first day, use this same scaling for other days (#scaler)
    scaler = StandardScaler().fit (df_day0_pivot[feat_cols])

    pca_result = run_PCA (df_day0_pivot, scaler, pca, feat_cols, pca_num_components, True)

    loadings = pd.DataFrame (pca.components_.T, columns = ["PC{}".format (x) for x in range (1,pca_num_components+1)], \
                             index = feat_cols)
    plot_loadings  (pca_result[:,0:2], pca.components_.T, feat_cols, "loadings.png")
   # print (loadings)
    print ("Df day 0")
    print (df_day0_pivot.columns)
    print (df_day0_pivot.head())
    outfig = os.path.join (outdir, "PCA")
  #  df_day0_pivot = add_age_gender(df_day0_pivot,xls_file)

    print (df_day0_pivot.head())
    plot_PCA (df_day0_pivot, outfig)
    print_PC_correlates (df_day0_pivot, pca_num_components, os.path.join (outdir, "PCA.tsv"))
#    tsne_results = run_TSNE(df_day0_pivot, pca_result)
#    plot_TSNE(df_day0_pivot, tsne_results, os.path.join (outdir, "TSNE.png"))

    # day 54 missing a lot of individuals? only 6 subjects, instead of 40
   # df_day54 = df_day54[feat_cols]
    df_day54_pivot= apply_pca (df_day54, scaler, pca, pca_num_components, feat_cols, xls_file) #GHB and GHB2 measurements missing
    #print (df_day54_pivot.head())
    #exit (0)
    #df_day96 = df_day96[feat_cols]
    df_day96_pivot = apply_pca (df_day96, scaler, pca, pca_num_components, feat_cols, xls_file)
    #print(df_day0_pivot['SUBJECT'].nunique())
    #print(df_day54_pivot['SUBJECT'].nunique())

  #  print (df_day96_pivot.head())
   # print (df_day96_pivot.columns)
    #print (df_day0_pivot.shape)
    #print (df_day0_pivot['SUBJECT'].unique)
    #print(df_day54_pivot.shape)
    #print (df_day54_pivot['SUBJECT'].unique)
    #print(df_day96_pivot.shape)
    #exit (0)
    plot_trajectories ([df_day0_pivot,  df_day54_pivot, df_day96_pivot], os.path.join (outdir, "PCA_trajectories.png"))