import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
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

def run_PCA (df, pca, feat_cols, pca_components, fit_transform = True):
    print ('running PCA')

    autoscaler = StandardScaler()
    autoscaled_df = autoscaler.fit_transform (df[feat_cols])
    num_components = pca_components

   # print (df[feat_cols].values)
   # pca_result = pca.fit_transform (df[feat_cols].values)
    if fit_transform:
        pca_result = pca.fit_transform(autoscaled_df)
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

    # format should be
    x_array = [1,2,3]
    y_array = [2,4,8]
    z_array = [3,6,10]
    #3D

    first_df = df_list[0]
   # print (first_df.columns)
    pc_df = first_df[['SUBJECT','PCA1', 'PCA2','PCA3']]
    for df_index in range (1, len(df_list)):
        df = df_list[df_index]
        pc_df = pc_df.merge (df[['SUBJECT', 'PCA1', 'PCA2', 'PCA3']], on="SUBJECT", how='left')
    #    print (pc_df.head())
 #       x_array.append (df['PC1'])
 #       y_array.append (df['PC2'])
 #       z_array.append (df['PC3'])
    ax = plt.figure (figsize=(16,10)).gca (projection='3d')

    pca1_columns = pc_df.columns[pc_df.columns.str.contains ("PCA1")]
    pca2_columns = [col for col in pc_df.columns if col.startswith ("PCA2")]
    pca3_columns = [col for col in pc_df.columns if col.startswith("PCA3")]

    PC1_array = pc_df[pca1_columns].to_numpy()
    PC2_array = pc_df[pca2_columns].to_numpy()
    PC3_array = pc_df[pca3_columns].to_numpy()
    print (pca1_columns)
  #  for sample in df:
    for sample_index in range (0, len(PC1_array)):
        ax.plot (PC1_array[sample_index],PC2_array[sample_index],PC3_array[sample_index]) # need to assign color c=colors
    ax.set_xlabel ('PC1')
    ax.set_ylabel ('PC2')
    ax.set_zlabel ('PC3')
    #, data=df)
    #hue="SUBJECT", \
    #palette=sns.color_palette('hls', len(df.index)), data=df, legend="full", alpha=0.3)
    plt.savefig (outfig)
    plt.clf()

def run_TSNE (df, pca_result):
    tsne = TSNE (n_components=2, verbose=0, perplexity=40, n_iter=300)
    tsne_pca_results = tsne.fit_transform (pca_result)
    return tsne_pca_results

def print_PC_correlates (df, pca_num_components, pca_results_file):
    corr = df.corr()
    corr_abs = corr.abs()
    s = corr_abs.unstack ()
    so = s.sort_values (kind="quicksort", ascending=False)
 #   print (type (so))
    so = so.to_frame().reset_index()
    so.columns = ['x','y','abs(corr)']
    print(so.head())
   # exit()
#    print (so[0])
    so = so[so['x'] != so['y']]
    so = so[so['abs(corr)'] >= 0.5]
    so = so[so['x'].str.contains ("PC")]
    so= so.sort_values (by=['x', 'abs(corr)'], ascending=[True, False])
    print (so.head())
    so.to_csv (pca_results_file, sep="\t", index=False)


def apply_pca (df_day, pca, pca_num_components, feat_cols, xls_file):
    df_day_pivot = df_day.pivot('SUBJECT', 'TEST', 'RES')
    # df_day0_pivot['RES'].reset_index()
    df_day_pivot.reset_index()
    df_day_pivot.columns.name = None
    df_day_pivot = df_day_pivot.dropna()  # drop rows with NaN
    pca_result = run_PCA (df_day_pivot, pca, feat_cols, pca_num_components, False)
    df_day_pivot = add_age_gender(df_day_pivot,xls_file)
    return df_day_pivot

if __name__ == '__main__':
    xls_file = "/home/pauline/exercise_study_data/ExerciseStudy.xls"
    outdir = "/home/pauline/exercise_study/"
    df = read_metabolite_data(xls_file)
    df_day0 = df[df['Day'] == 0]
    df_day96 = df[df['Day'] == 96]
    df_day54 = df[df['Day'] == 54]
    print (df['Day'].value_counts())
#    exit()
 #   print (df_day0.head())
    df_day0_pivot = df_day0.pivot ('SUBJECT', 'TEST', 'RES')
   # df_day0_pivot['RES'].reset_index()
    df_day0_pivot.reset_index()
    df_day0_pivot.columns.name=None
    df_day0_pivot = df_day0_pivot.dropna()  # drop rows with NaN

  #  df_day0_pivot.columns = df_day0_pivot.columns.droplevel().rename(None)
    print (df_day0_pivot.head())
    feat_cols = df_day0_pivot.columns.to_list()

    # TODO, comment below out once get new dataset. GHB and GHB2 missing for day 54
    feat_cols = [col  for col in feat_cols if col != 'GHB' and col != 'GHB2']
  #  print ("hi there")
  #  print (feat_cols)
#    feat_cols = [x for x in feat_cols if (x != "Age_group" & x != 'Age' & x != "Gender" & x != 'SUBJECT')]
 #   feat_cols = [x for x in feat_cols if x != "Age_group"]
  #  print ("hi 2")
   # print (feat_cols)
    pca_num_components = 8
    pca = PCA(n_components=pca_num_components)
    pca_result = run_PCA (df_day0_pivot, pca, feat_cols, pca_num_components, True)
    outfig = os.path.join (outdir, "PCA")
    df_day0_pivot = add_age_gender(df_day0_pivot,xls_file)

    print (df_day0_pivot.head())
    plot_PCA (df_day0_pivot, outfig)
    print_PC_correlates (df_day0_pivot, pca_num_components, os.path.join (outdir, "PCA.tsv"))
#    tsne_results = run_TSNE(df_day0_pivot, pca_result)
#    plot_TSNE(df_day0_pivot, tsne_results, os.path.join (outdir, "TSNE.png"))

    # day 54 missing a lot of individuals? only 6 subjects, instead of 40
    df_day54_pivot= apply_pca (df_day54, pca, pca_num_components, feat_cols, xls_file) #GHB and GHB2 measurements missing
    df_day96_pivot = apply_pca (df_day96, pca, pca_num_components, feat_cols, xls_file)
    print(df_day0_pivot['SUBJECT'].nunique())
    print(df_day54_pivot['SUBJECT'].nunique())

  #  print (df_day96_pivot.head())
   # print (df_day96_pivot.columns)
    print (df_day0_pivot.shape)
    print(df_day54_pivot.shape)
    print(df_day96_pivot.shape)
    plot_trajectories ([df_day0_pivot,  df_day96_pivot], os.path.join (outdir, "PCA_trajectories.png"))