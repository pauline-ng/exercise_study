import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
from datetime import datetime
from scipy.stats import linregress
matplotlib.use('TkAgg')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    outdir = "/home/pauline/exercise_study/"

    xl = pd.ExcelFile("/home/pauline/exercise_study/ExerciseStudy.xls")
    sheet_names = xl.sheet_names
    df_body = xl.parse ("Body measurements")



    df_clinical_lipids = xl.parse ("Clinical lipids")

    # add CRP sheet
    df_crp = xl.parse ("CRP")
    df_clinical_lipids =   df_clinical_lipids.append (df_crp)

    # add TSH sheet
    df_TSH = xl.parse ("TSH")
    df_clinical_lipids = df_clinical_lipids.append (df_TSH[['SUBJECT', 'DATE', 'TIME', 'TEST', 'RES', 'UNIT']])

    df_clinical_lipids.reset_index (drop=True)

    # drop rows where there's no data
    df_clinical_lipids = df_clinical_lipids[df_clinical_lipids['RES'].notna()]

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
        plt.plot (df.index, df['RES'])
    #    plt.imshow()
      #  plt.show (block=True)
        print (df.head())
        test = df["TEST"].iloc[0]

        avg_df = pd.DataFrame ({'RES': df.groupby (['SUBJECT','Day'])['RES'].mean()}).reset_index()

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
        lin_reg.to_csv (os.path.join (outdir, test + "linreg.tsv"), sep="\t", index=True)
        #print (lin_reg)
        sns.lmplot (data=avg_df,
                        x="Day",
                        y="RES",
                        hue="SUBJECT")
        #ax = plt.gca()
        #ax.set_title (df["TEST"].iloc[0])
        # remove legend, too many individuals
        plt.legend([], [], frameon=False)

        plt.title (test)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, test + ".png"))
        plt.close()

