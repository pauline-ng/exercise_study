import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
from datetime import datetime
from scipy.stats import linregress
from scipy.stats import spearmanr

def plot_hist (df, outdir_fig):
    x_array = []
    age_gender_groups = df['Age_gender'].unique()
    #print (df.head())
    for group in age_gender_groups:
        x = df[df['Age_gender'] == group]["slope"]
        x = x.to_list() #.to_list()
        #print (x)
        x_array.append (x)
    #print (x_array)
    plt.hist (x_array, label=age_gender_groups)
    plt.legend(age_gender_groups)
    plt.savefig (outdir_fig)
    plt.clf()

def plot_by_gender (avg_df, test, outdir_fig):
#    fig, ax = plt.subplots() # for overlay 2 plots
    sns.lmplot (data=avg_df[avg_df["Gender"] == "F" ],
                        x="Day",
                        y="RES",
                        hue="SUBJECT", line_kws={'color': 'red'}, ci=None, legend=False)
    plt.title(test + " females")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir_fig, test + "_F.png"))
    plt.close()
    plt.clf()

     #   ax2 = ax.twinx() # for overlay of 2 plots
    sns.lmplot (data=avg_df[avg_df["Gender"] == "M" ],
                        x="Day",
                        y="RES",
                        hue="SUBJECT", line_kws={'color': 'blue'}, ci=None, legend=False)

    plt.title(test + " males")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir_fig, test + "_M.png"))
    plt.close()
    plt.clf()

    exit()

# field can be slope if discarding outliers by linear regression
# field can be RES if discarding single points
def discard_outliers (df, field, outfile):

    plt.figure ()
    plt.subplot (1,2,1)  # 1 row, 2 columns, 1rst plot
    mean = df[field].mean()
    stddev = df[field].std()
    xmin = df[field].min()
    xmax = df[field].max()
    plt.hist (df[field], bins=10, range=[xmin, xmax])

    rows_to_drop = df[(df[field] > mean + 2 * stddev) | (df[field] < mean - 2 * stddev)]
    rows_to_drop.to_csv(outfile + "dropped.txt", sep="\t",index=False )
    df = df.drop (rows_to_drop.index)
    df = df.reindex()
#    df = df.drop (df[df['slope'] > slope_mean + 2 * slope_stddev].index)
#    df = df.drop(df[df['slope'] < slope_mean - 2 * slope_stddev].index)

    plt.subplot (1,2,2)  # 1 row, 2 columns, 1rst plot
    plt.hist (df[field], bins=10, range=[xmin, xmax] )
    #plt.legend()

    plt.savefig (outfile)
    plt.clf()

    return df


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    outdir = "/home/pauline/exercise_study/"

    xl = pd.ExcelFile("/home/pauline/exercise_study_data/ExerciseStudy.xls")
    sheet_names = xl.sheet_names
    df_body = xl.parse ("Body measurements")
    df_age_gender = df_body[['Subject', 'Age', 'Gender']].drop_duplicates()
    df_age_gender = df_age_gender.rename (columns={'Subject': 'SUBJECT'})
    df_age_gender['age_group'] = df_age_gender.apply (lambda row: 'Under50' if row['Age'] < 50 else 'Over50', axis=1)
    df_age_gender['Age_gender'] = df_age_gender.apply (lambda row: ' '.join ([row['Gender'], row['age_group']]), axis=1)
    print (df_age_gender.head())


    df_clinical_lipids = xl.parse ("Clinical lipids")

    # add CRP sheet
    df_crp = xl.parse ("CRP")
    df_clinical_lipids =   df_clinical_lipids.append (df_crp)

    # add TSH sheet
    df_TSH = xl.parse ("TSH")
    df_clinical_lipids = df_clinical_lipids.append (df_TSH[['SUBJECT', 'DATE', 'TIME', 'TEST', 'RES', 'UNIT']])

    # add FBC sheet
    df_FBC = xl.parse ("FBC")
    df_clinical_lipids = df_clinical_lipids.append(df_FBC[['SUBJECT','DATE', 'TIME', 'TEST', 'RES','UNIT']])
    df_clinical_lipids.reset_index (drop=True)

    # drop rows where there's no data
    df_clinical_lipids = df_clinical_lipids[df_clinical_lipids['RES'].notna()]

    # drop GHB and GHB2, because there are only 2 timepoints
    #df_clinical_lipids = df_clinical_lipids[~df_clinical_lipids['TEST'].str.contains('GHB', 'GHB2')]
    #df_clinical_lipids.reset_index(drop=True)

    # turn Date into integer to allow for linear regression
    df_clinical_lipids["Day"] = df_clinical_lipids.apply(lambda row: datetime.strptime (row["DATE"],'%d/%m/%Y').timetuple().tm_yday, axis=1)
    df_clinical_lipids["Day"] = df_clinical_lipids["Day"].astype (int)

    # clean out non-numeric entries
    df_clinical_lipids = df_clinical_lipids[df_clinical_lipids['RES'].apply(lambda x: type(x) in [int, np.int64, float, np.float64])]


    # get first day, then normalize to that
    first_day = df_clinical_lipids["Day"].min()
    df_clinical_lipids["Day"] = df_clinical_lipids["Day"] - first_day
    grouped = df_clinical_lipids.groupby (["TEST"])
    dfs = [x for _, x in df_clinical_lipids.groupby ('TEST')]
    for df in dfs:
        test = df["TEST"].iloc[0]
        print (test)

      # now convert it back to float
        df['RES'] = df['RES'].astype (float)
#        plt.plot (df.index, df['RES'])
        test = df["TEST"].iloc[0]

        avg_df = pd.DataFrame ({'RES': df.groupby (['SUBJECT','Day'])['RES'].mean()}).reset_index()
        avg_df = discard_outliers (avg_df, 'RES', os.path.join (outdir, test + ".rmoutliers_from_res.png"))
        avg_df = avg_df.merge(df_age_gender, on="SUBJECT", how='left')

        sample_groups = avg_df.groupby ('SUBJECT')
        lin_reg = (sample_groups.apply (lambda x: pd.Series (linregress (x['Day'], x['RES'])))
            .rename(columns={
                0: 'slope',
                1: 'intercept',
                2: 'rvalue',
                3: 'pvalue',
                4: 'stderr'
            })
        )
   #     print (lin_reg.head())
     #   print (df_age_gender)
#        lin_reg = lin_reg.rename_axis ('SUBJECT').reset_index()
        lin_reg['SUBJECT'] = lin_reg.index
        lin_reg = lin_reg.rename_axis(None)
        lin_reg = lin_reg.reset_index(drop=True)
       # print (lin_reg.head())
 #       lin_reg = lin_reg.reset_index()
        lin_reg = lin_reg.merge(df_age_gender, on="SUBJECT", how='left')
        lin_reg_male = lin_reg[lin_reg['Gender'] == 'M']
        spearman_male_coef, spearman_male_p = spearmanr (lin_reg_male['Age'], lin_reg_male['slope'])
        lin_reg_female = lin_reg[lin_reg['Gender'] == 'F']
        spearman_female_coef, spearman_female_p = spearmanr(lin_reg_female['Age'], lin_reg_female['slope'])
      #  print (str(spearman_female_coef))

        lin_reg = discard_outliers (lin_reg,'slope', os.path.join (outdir, test + "_dist.png" ) )
        lin_reg_outfile = os.path.join (outdir, test + "linreg.tsv")
        lin_reg.to_csv (lin_reg_outfile, sep="\t") #, index=True)
        plot_hist(lin_reg,  os.path.join (outdir, test + "_hist.png"))

        with open (lin_reg_outfile, "a") as outfp:
            outfp.write (str(lin_reg.mean()))

            # 3 results, expect 3 by chance (30*2*0.05 = 3)
            if spearman_male_p < 0.05:
                outfp.write ("\t".join ([test, "Spearman male", str(spearman_male_coef), str(spearman_male_p)]) + '\n')
            if spearman_female_p < 0.05:
                outfp.write("\t".join ([test, "Spearman female", str(spearman_female_coef), str(spearman_female_p)]) + '\n')


        gender_means = lin_reg.groupby (['Gender']).mean()
        age_means =lin_reg.groupby (['age_group']).mean()
        gender_age = lin_reg.groupby (['Gender', 'age_group']).mean()
        #print (lin_reg)
        gender_means.to_csv (os.path.join (outdir, test + "linreg_gender.tsv"), sep="\t", index=True)
        age_means.to_csv (os.path.join (outdir, test + "linreg_age.tsv"), sep="\t", index=True)
        gender_age.to_csv (os.path.join (outdir, test + "linreg_age_gender.tsv"), sep="\t", index=True)
      #  print (age_means)

        outdir_fig = os.path.join (outdir, "figures/")
        if not os.path.exists (outdir_fig):
            os.makedirs (outdir_fig)

    #    fig, ax = plt.subplots() # for overlay 2 plots
        sns.lmplot (data=avg_df,
                        x="Day",
                        y="RES",
                        hue="SUBJECT", line_kws={'color': 'black'}, ci=None, legend=False)
        plt.title(test)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir_fig, test + ".png"))
        plt.close()
        plt.clf()



