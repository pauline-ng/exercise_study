import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.spines as spines
import matplotlib.ticker as ticker

import os
import numpy as np
import seaborn as sns
from datetime import datetime
from scipy.stats import linregress
matplotlib.use('TkAgg')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    outdir = "/home/pauline/exercise_study_data/"

    xl = pd.ExcelFile("/home/pauline/exercise_study_data/ExerciseStudy.xls")
    sheet_names = xl.sheet_names
    df_body = xl.parse ("Body measurements")

    df_body.rename (columns = {"Value: 09 Sep": "Day 1", "Value: 02 Nov": "Day 60"}, inplace=True)
    # drop rows where Day 60 isn't observed
    df_body = df_body[df_body['Day 60'].notna()]

    # clean out non-numeric entries
    df_body = df_body[df_body['Day 1'].apply(lambda x: type(x) in [int, np.int64, float, np.float64])]
    df_body = df_body[df_body['Day 60'].apply(lambda x: type(x) in [int, np.int64, float, np.float64])]

    print (df_body.head())
    df_body["Change"] = df_body.apply (lambda row: row["Day 60"]- row["Day 1"], axis=1)
    print(df_body.head())


    mean_change = pd.DataFrame ({'Mean Change': df_body.groupby (["Measurement parameter"])['Change'].mean()}).reset_index()
    std_dev = pd.DataFrame ({'Std Dev': df_body.groupby (["Measurement parameter"])['Change'].std()}).reset_index()
    print (mean_change)
    print (std_dev)

# plate data
    ax = df_body.hist(column='Change', by='Measurement parameter', bins=10, grid=False, figsize=(8, 10), layout=(5, 2),
                 sharex=True, color='#86bf91', zorder=2, rwidth=0.9)
    ax = ax.flatten()
    for i, x in enumerate(ax):

        # Despine
        x.spines['right'].set_visible(False)
        x.spines['top'].set_visible(False)
        x.spines['left'].set_visible(False)

        # Switch off ticks
        x.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off",
                      labelleft="on")

        # Draw horizontal axis lines
        vals = x.get_yticks()
        for tick in vals:
            x.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

        # Set x-axis label
        x.set_xlabel("Measurement", labelpad=20, weight='bold', size=12)

        # Set y-axis label
        if i == 1:
            x.set_ylabel("Sessions", labelpad=50, weight='bold', size=12)

            # Format y-axis label
        x.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,g}'))

        x.tick_params(axis='x', rotation=0)
    plt.show()
    exit(0)
    body_tests = df_body.groupby (["Measurement parameter"])['Change']
    dfs = [x for _, x in body_tests]
    for df in dfs:
#        df['Change'].plot.hist (bins=10)
        df.hist (column="Change")

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

