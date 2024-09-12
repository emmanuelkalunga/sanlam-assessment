import marimo

__generated_with = "0.8.2"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import numpy as np
    return mo, np, pd


@app.cell
def __(pd):
    raw_data = pd.read_csv('conversion_data.csv')
    return raw_data,


@app.cell
def __(mo):
    def center_bold_text(text):
        return mo.md(f"<div style='text-align: center;'><b>{text}</div>")
    return center_bold_text,


@app.cell
def __(mo):
    mo.md(r"""# EDA""")
    return


@app.cell
def __(raw_data):
    raw_data.describe()
    return


@app.cell
def __(mo):
    mo.md(r"""## Sankey Chart""")
    return


@app.cell
def __(pd):
    # Gnerate Sankey data
    def generate_sankey_chart_data(df: pd.Dataframe, columns: list, sankey_link_weight: str):
        # list of list: each list are the set of nodes in each tier/column
        column_values = [df[col] for col in columns]

        # this generates the labels for the sankey by taking all the unique values
        labels = sum([list(node_values.unique()) for node_values in column_values],[])

        # initializes a dict of dicts (one dict per tier) 
        link_mappings = {col: {} for col in columns}

        # each dict maps a node to unique number value (same node in different tiers
        # will have different nubmer values
        i = 0
        for col, nodes in zip(columns, column_values):
            for node in nodes.unique():
                link_mappings[col][node] = i
                i = i + 1

        # specifying which coluns are serving as sources and which as sources
        # ie: given 3 df columns (col1 is a source to col2, col2 is target to col1 and 
        # a source to col 3 and col3 is a target to col2
        source_nodes = column_values[: len(columns) - 1]
        target_nodes = column_values[1:]
        source_cols = columns[: len(columns) - 1]
        target_cols = columns[1:]
        links = []

        # loop to create a list of links in the format [((src,tgt),wt),(),()...]
        for source, target, source_col, target_col in zip(source_nodes, target_nodes, source_cols, target_cols):
            for val1, val2, link_weight in zip(source, target, df[sankey_link_weight]):
                links.append(
                    (
                        (
                            link_mappings[source_col][val1],
                            link_mappings[target_col][val2]
           ),
           link_weight,
        )
                 )

        # creating a dataframe with 2 columns: for the links (src, tgt) and weights
        df_links = pd.DataFrame(links, columns=["link", "weight"])

        # aggregating the same links into a single link (by weight)
        df_links = df_links.groupby(by=["link"], as_index=False).agg({"weight": sum})

        # generating three lists needed for the sankey visual
        sources = [val[0] for val in df_links["link"]]
        targets = [val[1] for val in df_links["link"]]
        weights = df_links["weight"]

        return labels, sources, targets, weights
    return generate_sankey_chart_data,


@app.cell
def __(raw_data):
    raw_data.columns
    return


@app.cell
def __(raw_data):
    sankey_df_old = raw_data.assign(
        clicks_bin = lambda x: x.Clicks.apply(lambda y: 'Clicks' if y > 0 else 'No Clicks')
    )
    return sankey_df_old,


@app.cell
def __(np, raw_data):
    sankey_df = raw_data.assign(
        clicks_bin = lambda x: np.where(x.Clicks > 0, 'Clicks', 'No Clicks'),
        weight = 1,
        charge = lambda x: np.where(x.Spent > 0, 'Spend', 'No Spend'),
        TC = lambda x: x.Total_Conversion.apply(lambda y: ": ".join(["TC", str(y)])),
        AC = lambda x: x.Approved_Conversion.apply(lambda y: ": ".join(["AC", str(y)])),
        interest_new = lambda x: x.interest.apply(lambda y: ": ".join(["Interest group", str(y)])),
        age_new = lambda x: x.age.apply(lambda y: ": ".join(["Age", str(y)])),
        gender_new = lambda x: x.gender.apply(lambda y: ": ".join(["Gender", str(y)]))

    )
    return sankey_df,


@app.cell
def __():
    import plotly.graph_objects as go
    return go,


@app.cell
def __(generate_sankey_chart_data, go, sankey_df):
    # flow campaign: gender -> age -> interest -> clicks
    labels, sources, targets, weights = generate_sankey_chart_data(df=sankey_df,
                                                                   columns=['gender_new', 'age_new','interest_new', 'clicks_bin'],
                                                                   sankey_link_weight='weight') 
    fig = go.Figure(data=[go.Sankey(
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(color = "black", width = 0.5),
          label = labels,
          color = "blue"
        ),
        link = dict(
          source = sources, # indices correspond to labels, eg A1, A2, A1, B1, ...
          target = targets,
          value = weights
      ))])

    fig.update_layout(title_text="Flow campaign: gender -> age -> interest -> clicks", font_size=10)
    fig.show()
    return fig, labels, sources, targets, weights


@app.cell
def __(generate_sankey_chart_data, go, sankey_df):
    # Flow campaign spend: gender -> age -> interest -> clicks (True/False)
    labels_s, sources_s, targets_s, weights_s = generate_sankey_chart_data(df=sankey_df,
                                                                   columns=['gender_new', 'age_new','interest_new', 'clicks_bin'],
                                                                   sankey_link_weight='Spent') 
    fig_s = go.Figure(data=[go.Sankey(
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(color = "black", width = 0.5),
          label = labels_s,
          color = "blue"
        ),
        link = dict(
          source = sources_s, # indices correspond to labels, eg A1, A2, A1, B1, ...
          target = targets_s,
          value = weights_s
      ))])

    fig_s.update_layout(title_text="Flow campaign spend: gender -> age -> interest -> clicks", font_size=10)
    fig_s.show()
    return fig_s, labels_s, sources_s, targets_s, weights_s


@app.cell
def __(generate_sankey_chart_data, go, sankey_df):
    # Full funnel
    labels_ac, sources_ac, targets_ac, weights_ac = generate_sankey_chart_data(df=sankey_df,
                                                                   columns=['gender_new', 'age_new', 'interest_new', 'clicks_bin', 'charge', 'TC', 'AC'],
                                                                   sankey_link_weight='weight') 

    fig_ac = go.Figure(data=[go.Sankey(
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(color = "black", width = 0.5),
          label = labels_ac,
          color = "blue"
        ),
        link = dict(
          source = sources_ac, # indices correspond to labels, eg A1, A2, A1, B1, ...
          target = targets_ac,
          value = weights_ac
      ))])

    fig_ac.update_layout(title_text="Full Funnel", font_size=10)
    fig_ac.show()
    return fig_ac, labels_ac, sources_ac, targets_ac, weights_ac


@app.cell
def __(raw_data):
    raw_data.columns
    return


@app.cell
def __(raw_data):
    raw_data[raw_data.Clicks==0]['Approved_Conversion'].value_counts()
    return


@app.cell
def __(raw_data):
    raw_data[raw_data.Clicks==0]['Total_Conversion'].value_counts()
    return


@app.cell
def __():
    return


@app.cell
def __(mo):
    mo.md(r"""## Understanding Variables""")
    return


@app.cell
def __():
    import seaborn as sns
    import matplotlib.pyplot as plt
    from ydata_profiling import ProfileReport
    import warnings
    warnings.filterwarnings('ignore')
    return ProfileReport, plt, sns, warnings


@app.cell
def __(raw_data):
    raw_data.info()
    return


@app.cell
def __(raw_data):
    # Any missing data?
    raw_data.describe(include='all').T
    return


@app.cell
def __(raw_data):
    raw_data.isnull().sum()
    return


@app.cell
def __(ProfileReport, raw_data):
    # view the analysis result inside jupyter
    prof = ProfileReport(raw_data,  minimal=False, title="Variables Analysis")
    return prof,


@app.cell
def __(prof):
    prof
    return


@app.cell
def __(prof):
    # save the analysis to html
    prof.to_file(output_file='variables_analysis.html')
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Variable Distributions
        We will look into the charateristics of individual variables.
        """
    )
    return


@app.cell
def __(mo):
    mo.md("""#### Setting the right data types""")
    return


@app.cell
def __(raw_data):
    raw_data.info()
    return


@app.cell
def __(raw_data):
    processed_data = raw_data.copy()
    processed_data = processed_data.assign(
        ad_id = lambda x: x["ad_id"].astype("category"),
        xyz_campaign_id = lambda x: x["xyz_campaign_id"].astype("category"),
        fb_campaign_id = lambda x: x["fb_campaign_id"].astype("category"),
        age = lambda x: x["age"].astype("category"),
        gender = lambda x: x["gender"].astype("category"),
        interest = lambda x: x["interest"].astype("category")    
    )
    return processed_data,


@app.cell
def __(processed_data):
    processed_data.info()
    return


@app.cell
def __(ProfileReport, processed_data):
    # view the analysis result inside jupyter
    prof_processed = ProfileReport(processed_data,  minimal=False, title="Processed Variables Analysis")
    # save the analysis to html
    prof_processed.to_file(output_file='processed_variables_analysis.html')
    return prof_processed,


@app.cell
def __(mo):
    mo.md(r"""#### Analyse Categorical variables""")
    return


@app.cell
def __(mo, processed_data):
    # Identify categorical columns
    categorical_columns = processed_data.select_dtypes(include=['category']).columns
    mo.md(f"""Categorical columns are: {list(categorical_columns)}""")
    return categorical_columns,


@app.cell
def __(categorical_columns, plt, processed_data, sns):
    # Function to create barplots that indicate percentage for each category.

    def perc_on_bar(plot, feature):
        '''
        plot
        feature: categorical feature
        the function won't work if a column is passed in hue parameter
        '''
        total = len(feature) # length of the column
        for p in ax.patches:
            percentage = '{:.1f}%'.format(100 * p.get_width()/total) # percentage of each class of the category
            x = p.get_x() + p.get_width() # width of the plot
            y = p.get_y() + p.get_height() /2 -0.05           # hieght of the plot
            ax.annotate(percentage, (x, y), size = 12) # annotate the percantage 
        plt.show() # show the plot


    # Iterate through each categorical column
    for col in list(set(categorical_columns) - set(['ad_id', 'fb_campaign_id'])):
        plt.figure(figsize=(10,5))
        ax = sns.countplot(processed_data[col],palette='winter')
        # perc_on_bar(ax,processed_data[col])
        perc_on_bar(ax, processed_data[col])
    return ax, col, perc_on_bar


@app.cell
def __(mo):
    mo.md(r"""#### Analyse numerical variables""")
    return


@app.cell
def __(mo, processed_data):
    numerical_columns = processed_data.select_dtypes(exclude=['category']).columns
    mo.md(f"""Numerical columns are: {list(numerical_columns)}""")
    return numerical_columns,


@app.cell
def __(numerical_columns, plt, processed_data, sns):
    for col1 in numerical_columns:
        plt.figure(figsize=(10,1))
        sns.displot(processed_data, x = col1, height=5)
        plt.show()

    # plt.figure(figsize=(5,5))
    # sns.displot(processed_data, x = "Impressions", height=5)
    # plt.show()
    return col1,


@app.cell
def __(mo):
    mo.md(
        r"""
        #### Conslusion
        * Targetting more the 30-34 age range: 37.3%.
        * Large imbalance in xyz campaigns. 
        * Interest groups largely varied.
        * Genders are balanced. 
        * The numerical variable (i.e. 'Impressions', 'Clicks', 'Spent', 'Total_Conversion',
               'Approved_Conversion') all follow an gamma distribution (exponential decay)
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Variable interactions and correlation
        Plot correlations and pairplot
        """
    )
    return


@app.cell
def __(plt, processed_data, sns):
    plt.figure(figsize=(18,6))
    sns.heatmap(processed_data.corr(numeric_only=True), annot=True, annot_kws={"size": 14})
    plt.xticks(fontsize=14)  # Adjust the font size for x-axis labels
    plt.yticks(fontsize=14)  # Adjust the font size for y-axis labels

    plt.show()
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        #### Observations
        * Very high correlation (almost 1) between spent and clicks. This suggest that this is a paid per click campaign.
        * There is a high correlation between impressions & clicks. While correlation does not account for causation, it can be said here that more impressions lead to more clicks.
        * Generally, there is, as could be expected, a high correlation between imoprssions, clicks, spend, and total conversions.
        """
    )
    return


@app.cell
def __(processed_data):
    processed_data.columns
    return


@app.cell
def __(plt, processed_data, sns):
    sns.pairplot(data=processed_data.drop(["ad_id", "fb_campaign_id"],axis=1)) # ,hue="Product"
    plt.show()
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        #### Observations
        Strong correlations and colinearity bewtween variables `Impressions`, `Clicks`, `Spent`. With linearity also observed aith `Toatal_Conversion` and `Approved_Conversion`.
        """
    )
    return


@app.cell
def __(categorical_columns):
    categorical_columns
    return


@app.cell
def __(plt, processed_data, sns):
    sns.pairplot(data=processed_data, hue="xyz_campaign_id", plot_kws={'alpha': 0.5}, palette='bright') # ,hue="Product"
    plt.show()
    return


@app.cell
def __(processed_data):
    processed_data[["Impressions", "Clicks"]]
    return


@app.cell
def __(plt, processed_data, sns):
    sns.pairplot(data=processed_data[['xyz_campaign_id','Impressions', 'Clicks', 'Total_Conversion']], hue="xyz_campaign_id", plot_kws={'alpha': 0.5}, palette='bright', aspect=1.5) # ,hue="Product"
    plt.show()
    return


@app.cell
def __(mo):
    mo.md(r"""Similar behaviour accross campaign (i.e. `xyz_campaign_id`)""")
    return


@app.cell
def __(plt, processed_data, sns):
    sns.pairplot(data=processed_data, hue="age", plot_kws={'alpha': 0.4}, palette='bright') # ,hue="Product"
    plt.show()
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        #### Observation. 
        The correlation/interaction is stronger in some age groups than others. Examples:  

        * Age group 45-49 has a steeper correlation between impressions and clicks.  
        * Age group 30-34 has a steeper correlation between clicks and total conversion.
        """
    )
    return


@app.cell
def __(plt, processed_data, sns):
    sns.pairplot(data=processed_data, hue="gender", plot_kws={'alpha': 0.5}, palette='bright') # ,hue="Product"
    plt.show()
    return


@app.cell
def __(mo):
    mo.md(r"""Similarily, there is more clicks per impression for female compared to male.""")
    return


@app.cell
def __(plt, processed_data, sns):
    sns.pairplot(data=processed_data, hue="interest", plot_kws={'alpha': 0.5}, palette='bright') # ,hue="Product"
    plt.show()
    return


@app.cell
def __(processed_data):
    processed_data.columns
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        #### Total conversion per age and interest group

        Some interest groups have consistently higher conversion regardless of the age. The yunger age group shows more variations in conversion within interest groups
        """
    )
    return


@app.cell
def __(plt, processed_data, sns):
    plt.figure(figsize=(15,7))
    sns.boxplot(data=processed_data,
                x = "age",
                y = "Total_Conversion",
                hue="interest"
               )
    plt.legend(bbox_to_anchor=(1.00, 1))
    plt.show()
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        #### Conversion by campaign and interest group

        Some campaigns have more impressions and more conversion. It wil be insightful too look at the ratios such as click-through-rate, and impression to conversion.
        """
    )
    return


@app.cell
def __(plt, processed_data, sns):
    plt.figure(figsize=(18,6))
    sns.barplot(data=processed_data,
                x = "interest",
                y = "Total_Conversion",
                hue= "xyz_campaign_id"
               )
    # plt.legend(bbox_to_anchor=(1.00, 1))
    plt.show()
    return


@app.cell
def __(processed_data):
    processed_data.to_csv("processed_data.csv")
    return


@app.cell
def __(mo):
    mo.md(r"""#### Total Approved conversion per interest group""")
    return


@app.cell
def __(plt, processed_data, sns):
    # Group by "interest" and sum the "Approved_Conversion"
    grouped_processed_data = processed_data.groupby('interest')['Approved_Conversion'].sum().reset_index()
    # Plot the bar chart
    plt.figure(figsize=(18, 6))
    sns.barplot(x='interest', y='Approved_Conversion', data=grouped_processed_data)

    # Set the labels and title
    plt.xlabel('Interest')
    plt.ylabel('Total Approved Conversion')
    plt.title('Total Approved Conversion by Interest')

    # Rotate x-axis labels if needed
    plt.xticks(rotation=45)

    plt.show()
    return grouped_processed_data,


@app.cell
def __(mo):
    mo.md(r"""#### Approved conversions per age group""")
    return


@app.cell
def __(plt, processed_data, sns):
    # Group by "age" and sum the "Approved_Conversion"
    grouped_processed_data_age = processed_data.groupby('age')['Approved_Conversion'].sum().reset_index()
    # Plot the bar chart
    plt.figure(figsize=(18, 6))
    sns.barplot(x='age', y='Approved_Conversion', data=grouped_processed_data_age)

    # Set the labels and title
    plt.xlabel('Age')
    plt.ylabel('Total Approved Conversion')
    plt.title('Total Approved Conversion by Age')

    # Rotate x-axis labels if needed
    plt.xticks(rotation=45)

    plt.show()
    return grouped_processed_data_age,


@app.cell
def __(mo):
    mo.md(r"""#### Approved conversion per age group""")
    return


@app.cell
def __(plt, processed_data, sns):
    # Group by "gender" and sum the "Approved_Conversion"
    grouped_processed_data_gender = processed_data.groupby('gender')['Approved_Conversion'].sum().reset_index()
    # Plot the bar chart
    plt.figure(figsize=(18, 6))
    sns.barplot(x='gender', y='Approved_Conversion', data=grouped_processed_data_gender)

    # Set the labels and title
    plt.xlabel('Gender')
    plt.ylabel('Total Approved Conversion')
    plt.title('Total Approved Conversion by Gender')

    # Rotate x-axis labels if needed
    plt.xticks(rotation=45)

    plt.show()
    return grouped_processed_data_gender,


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Compaign performance analysis

        1. Evaluate the performance of each campaign by calculating
            1. Impressions, clicks, spent, conversions per campaign
            2. CTR, Cost-Per-Click (CPC)
            3. Conversion rates, VTC, CTC, cost per conversion (Amount spent per approved conversion)
            4. CTR i.e. Click-through rate (Ratio of clicks to impressions) for different demographics (age, gender, interest).
        3. Segmentation: Determine how different segments (age groups, gender, interests) are performing in terms of conversions and cost-effectiveness.
        4. Budget Allocation: Simulate different budget allocations to predict conversion outcomes based on historical data.
        """
    )
    return


@app.cell
def __():
    from enum import Enum
    return Enum,


@app.cell
def __(Enum):
    class Campaign(Enum):
        COMP1 = 916
        COMP2 = 936
        COMP3 = 1178
    return Campaign,


@app.cell
def __(Campaign):
    Campaign.COMP1.value
    return


@app.cell
def __(gp_campaign_ctr_cpc):
    gp_campaign_ctr_cpc
    return


@app.cell
def __(mo):
    mo.md(r"""### Impressions, clicks, spent, conversions per campaign""")
    return


@app.cell
def __(processed_data):
    gp_campaign = processed_data.groupby('xyz_campaign_id')[['Impressions', 'Clicks', 'Spent', 'Total_Conversion',
           'Approved_Conversion']].sum().reset_index()
    return gp_campaign,


@app.cell
def __(gp_campaign, pd, plt, sns):
    sns.set_palette("terrain") # nipy_spectral, hot, gnuplot2
    fig_camp, axes_camp = plt.subplots(1, 3, figsize=(20, 6))
    sns.barplot(data=gp_campaign, x='xyz_campaign_id', y='Impressions', ax=axes_camp[0])
    sns.barplot(x='xyz_campaign_id', y='value', hue='variable', 
                data=pd.melt(gp_campaign[['xyz_campaign_id', 'Clicks', 'Spent']], ['xyz_campaign_id']),
                ax=axes_camp[1]
               )
    sns.barplot(x='xyz_campaign_id', y='value', hue='variable', 
                data=pd.melt(gp_campaign[['xyz_campaign_id', 'Total_Conversion','Approved_Conversion']], ['xyz_campaign_id']),
                ax=axes_camp[2]
               )

    for ax_i in axes_camp:
        ax_i.tick_params(axis='both', labelsize=12)  # Adjust the font size for x-axis labels
        ax_i.xaxis.label.set_size(14) 
        ax_i.yaxis.label.set_size(14)
        ax_i.legend(fontsize=12)
    plt.suptitle("Impressions, clicks, spent, conversions by campaign", fontsize=16)
    plt.show()
    return ax_i, axes_camp, fig_camp


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Click-Through Rate (CTR) and Cost-Per-Click (CPC). 
        Percentage of people who clicked on campaign ad.  
        $CTR = \left( \frac{Number\ of\ Clicks}{Number\ of\ Impressions} \right) \times 100$
        """
    )
    return


@app.cell
def __(processed_data):
    processed_data
    return


@app.cell
def __(gp_campaign):
    gp_campaign_ctr_cpc = gp_campaign.assign(
        ctr = lambda x: 100*(x['Clicks']/x['Impressions']),
        cpc = lambda x: (x['Spent']/x['Clicks'])
    )
    return gp_campaign_ctr_cpc,


@app.cell
def __(gp_campaign_ctr_cpc, plt, sns):
    sns.set_palette("terrain") # nipy_spectral, hot, gnuplot2
    fig_camp_ctr_cpc, axes_camp_ctr_cpc = plt.subplots(1, 2, figsize=(20, 6))
    sns.barplot(data=gp_campaign_ctr_cpc, x='xyz_campaign_id', y='ctr', ax=axes_camp_ctr_cpc[0])
    sns.barplot(data=gp_campaign_ctr_cpc, x='xyz_campaign_id', y='cpc', ax=axes_camp_ctr_cpc[1])

    for ax_ii in axes_camp_ctr_cpc:
        ax_ii.tick_params(axis='both', labelsize=12)  # Adjust the font size for x-axis labels
        ax_ii.xaxis.label.set_size(14) 
        ax_ii.yaxis.label.set_size(16)
        # ax_ii.legend(fontsize=12)
    plt.suptitle("CTR & CPC", fontsize=16)
    plt.show()
    return ax_ii, axes_camp_ctr_cpc, fig_camp_ctr_cpc


@app.cell
def __(gp_campaign_ctr_cpc):
    gp_campaign_ctr_cpc[['xyz_campaign_id', 'ctr', 'cpc']]
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        #### Observations. 
        1. CTR (Click-Through Rate):
        	* Campaign `936` has the highest CTR (`0.0244%`), followed by Campaign `916` (`0.0234%`). Campaign `1178` has the lowest CTR (`0.0176%`).
        2. CPC (Cost per Click)
            * Campaign `916` has the lowest CPC (`1.32`), followd by Campaign `936` (`1.45`). Campaign `1178` has the highest CPC (`1.54%`).
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Conversion rate (CR). 

        \[Conversion\ Rate = 100 \times \frac{Conversions}{Clicks}\]
        """
    )
    return


@app.cell
def __(gp_campaign_ctr_cpc):
    gp_campaign_ctr_cpc_cr = gp_campaign_ctr_cpc.assign(
        total_cr = lambda x: 100*x['Total_Conversion']/x['Clicks'],
        approved_cr = lambda x: 100*x['Approved_Conversion']/x['Clicks'],
        cost_per_conversion = lambda x: x['Spent']/x['Approved_Conversion']
    )
    return gp_campaign_ctr_cpc_cr,


@app.cell
def __(gp_campaign_ctr_cpc_cr, plt, sns):
    # Example data for three campaigns
    campaigns = gp_campaign_ctr_cpc_cr['xyz_campaign_id']# ['Campaign 1', 'Campaign 2', 'Campaign 3']
    ctr = gp_campaign_ctr_cpc_cr['ctr'] # [5, 4, 6]  # Click-Through Rate (%)
    total_conv_rate = gp_campaign_ctr_cpc_cr['total_cr'] # [3, 2.5, 4]  # Total Conversion Rate (%)
    approved_conv_rate = gp_campaign_ctr_cpc_cr['approved_cr'] # [1.5, 1.2, 2]  # Approved Conversion Rate (%)

    # Create a figure
    fig_ft, axes_ft = plt.subplots(1, 3, figsize=(15, 6), sharey=True)

    # Plot funnel for each campaign
    for i, campaign in enumerate(campaigns):
        values = [ctr[i], total_conv_rate[i], approved_conv_rate[i]]
        stages = ['CTR', 'Total Conversion', 'Approved Conversion']

        axes_ft[i].barh(stages, values, color=[sns.color_palette('terrain')[3], sns.color_palette('terrain')[2], sns.color_palette('terrain')[1]])
        axes_ft[i].set_xlim(0, 80)
        axes_ft[i].set_title('Campaign '+ str(campaign))
        axes_ft[i].invert_yaxis()  # To have the funnel top-down

        for j in range(len(stages)):
            axes_ft[i].text(values[j] + 0.2, j, f'{values[j]:.3f}%', va='center', fontsize=14)

    # Set common labels
    fig_ft.text(0.5, -0.01, 'Percentage', ha='center', va='center', fontsize=14)
    # fig_ft.text(0.04, 0.5, 'Stages', ha='center', va='center', rotation='vertical', fontsize=14)

    plt.tight_layout()
    plt.show()
    return (
        approved_conv_rate,
        axes_ft,
        campaign,
        campaigns,
        ctr,
        fig_ft,
        i,
        j,
        stages,
        total_conv_rate,
        values,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        #### Observation. 

        Campaign 916 has the highest conversion rate (21.24%), which means it converts clicks into approved conversions more effectively. Campaign 936 has a conversion rate of 9.22%, while Campaign 1178 has a lower conversion rate (2.42%).
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Cost per Conversion

        Amount spent per conversion (approved)
        """
    )
    return


@app.cell
def __(gp_campaign_ctr_cpc_cr, pd, plt, sns):
    sns.set_palette("terrain") # nipy_spectral, hot, gnuplot2
    plt.figure(figsize=(12,6))
    sns.barplot(x='xyz_campaign_id', y='value', hue='variable', 
                data=pd.melt(gp_campaign_ctr_cpc_cr[['xyz_campaign_id', 'cpc', 'cost_per_conversion']], ['xyz_campaign_id'])
               )

    # for ax_i in axes_camp:
    #     ax_i.tick_params(axis='both', labelsize=12)  # Adjust the font size for x-axis labels
    #     ax_i.xaxis.label.set_size(14) 
    #     ax_i.yaxis.label.set_size(14)
    #     ax_i.legend(fontsize=12)
    plt.legend(fontsize=12)
    plt.xlabel('Campaign', fontsize=12)
    plt.ylabel('Cost', fontsize=12)
    plt.suptitle("CPC and Cost per Conversion", fontsize=16)
    plt.show()
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        * Campaign `916` has the lowest **CPC** (`1.32`), followd by Campaign `936` (`1.45`). Campaign `1178` has the highest CPC (`1.54`).
        * Campaign `916` has the lowest **Cost per Conversion** (`6.24`), followd by Campaign `936` (`15.81`). Campaign `1178` has the highest CPC (`63.83`).
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        # Simple What-if simulation

        Based on the compaign performance, give recommendations on where more money should be spent for better conversion
        """
    )
    return


@app.cell
def __(gp_campaign_ctr_cpc_cr, mo):
    mo.md(
        f"""
        **Total spend**: {gp_campaign_ctr_cpc_cr.Spent.sum()}.\n 
        **Total Approved Converstion**: {gp_campaign_ctr_cpc_cr.Approved_Conversion.sum()} 
        """
    )
    return


@app.cell
def __():
    CAMPAIGNS = [916, 936, 1178]
    return CAMPAIGNS,


@app.cell
def __(np, pd):
    def conversion(campaign_id: int, budget: float, campaign_properties: pd.DataFrame) -> dict:
        if campaign_id in list(campaign_properties.xyz_campaign_id):
            CPC = campaign_properties[campaign_properties['xyz_campaign_id']==campaign_id]['cpc'].iloc[0]
            CTR = campaign_properties[campaign_properties['xyz_campaign_id']==campaign_id]['ctr'].iloc[0]
            ACR = campaign_properties[campaign_properties['xyz_campaign_id']==campaign_id]['approved_cr'].iloc[0] # Approved Conversion Rate
            print(f"CPC: {CPC}, CTR: {CTR}, ACR: {ACR}")
            clicks = int(np.round(budget * (1/(CPC))))
            impressions = int(np.round(clicks * (100/CTR)))
            conversions = int(np.round(clicks * ACR/100))
            return dict(impressions = impressions, clicks = clicks, conversions = conversions) 
        else:
            raise Exception("Cannot simulate. Unknown Campaign ID")
    return conversion,


@app.cell
def __(mo):
    mo.md(
        """
        -------------------------------------
        ## Example Scenario (Campaign-based)
        """
    )
    return


@app.cell
def __(Campaign, conversion, gp_campaign_ctr_cpc_cr, mo):
    res1 = conversion(Campaign.COMP1.value, 1000, gp_campaign_ctr_cpc_cr)
    res2 = conversion(Campaign.COMP2.value, 1000, gp_campaign_ctr_cpc_cr)
    res3 = conversion(Campaign.COMP3.value, 1000, gp_campaign_ctr_cpc_cr)
    res_opt = conversion(Campaign.COMP1.value, gp_campaign_ctr_cpc_cr.Spent.sum(), gp_campaign_ctr_cpc_cr)

    mo.md(
        f"""
         **Total spend**: {gp_campaign_ctr_cpc_cr.Spent.sum()}.\n 
        **Total Impressions**: {gp_campaign_ctr_cpc_cr.Impressions.sum()} \n
        **Total Clicks**: {gp_campaign_ctr_cpc_cr.Clicks.sum()} \n
        **Total Approved Converstion**: {gp_campaign_ctr_cpc_cr.Approved_Conversion.sum()} \n
        - Spending 1000 on campaign {Campaign.COMP1.value} will result in {res1['impressions']} Impressions, {res1['clicks']} Clicks, and {res1['conversions']} Approved Conversions \n
        - Spending 1000 on campaign {Campaign.COMP2.value} will result in {res2['impressions']} Impressions, {res2['clicks']} Clicks, and {res2['conversions']} Approved Conversions \n
        - Spending 1000 on campaign {Campaign.COMP3.value} will result in {res3['impressions']} Impressions, {res3['clicks']} Clicks, and {res3['conversions']} Approved Conversions \n
        **Spending {gp_campaign_ctr_cpc_cr.Spent.sum()} on campaign {Campaign.COMP1.value} will result in**: \n
        - {res_opt['impressions']} Impressions, \n
        - {res_opt['clicks']} Clicks, and \n
        - {res_opt['conversions']} Approved Conversions
        """
         )
    return res1, res2, res3, res_opt


@app.cell
def __(mo):
    mo.md(
        r"""
        -------------------------------------
        ## Scenario based on target group
        """
    )
    return


@app.cell
def __(processed_data):
    gp_target = processed_data.groupby(['xyz_campaign_id', 'age', 'interest'])[['Impressions', 'Clicks', 'Spent', 'Total_Conversion', 'Approved_Conversion']].sum() #.reset_index()
    return gp_target,


@app.cell
def __(gp_target, np):
    gp_target_properties = gp_target.assign(
        ctr = lambda x: 100*(x['Clicks']/x['Impressions']),
        cpc = lambda x: (x['Spent']/x['Clicks']),
        total_cr = lambda x: 100*x['Total_Conversion']/x['Clicks'],
        approved_cr = lambda x: 100*x['Approved_Conversion']/x['Clicks'],
        cost_per_conversion = lambda x: x['Spent']/x['Approved_Conversion']
    )
    gp_target_properties = gp_target_properties.replace([np.inf, -np.inf], np.nan, inplace=False)
    gp_target_properties = gp_target_properties.dropna()
    return gp_target_properties,


@app.cell
def __(gp_target_properties):
    gp_target_properties_vis = gp_target_properties.reset_index()
    return gp_target_properties_vis,


@app.cell
def __(Campaign, gp_target_properties_vis, plt, sns):
    sns.set_palette("terrain") # nipy_spectral, hot, gnuplot2
    fig_x, axes_x = plt.subplots(1, 2, figsize=(20, 6))
    sns.barplot(data=gp_target_properties_vis[gp_target_properties_vis.xyz_campaign_id==Campaign.COMP3.value], x='interest', y='ctr', ax=axes_x[0])
    sns.barplot(data=gp_target_properties_vis[gp_target_properties_vis.xyz_campaign_id==Campaign.COMP3.value], x='interest', y='approved_cr', ax=axes_x[1])

    for ax_iii in axes_x:
        ax_iii.tick_params(axis='both', labelsize=12)  # Adjust the font size for x-axis labels
        ax_iii.xaxis.label.set_size(14) 
        ax_iii.yaxis.label.set_size(16)
        # ax_ii.legend(fontsize=12)
    plt.suptitle("CTR & CPC", fontsize=16)
    plt.show()
    return ax_iii, axes_x, fig_x


@app.cell
def __(mo):
    mo.md(
        r"""
        ----------
        ### Campaigns and Target Groups with **Highest CTR**
        """
    )
    return


@app.cell
def __(mo):
    top_ctr = mo.ui.slider(1, 30, label=f"Top CTR")
    return top_ctr,


@app.cell
def __(top_ctr):
    top_ctr
    return


@app.cell
def __(gp_target_properties, mo, top_ctr):
    top_target_ctr = gp_target_properties.sort_values('ctr', ascending=False).head(top_ctr.value).reset_index()[['xyz_campaign_id', 'age', 'interest', 'Impressions', 'Clicks', 'ctr']]
    mo.ui.table(
        data=top_target_ctr,
        # use pagination when your table has many rows
        pagination=False,
        label=f"""Top {top_ctr.value} campaigns and target groups (age & interest) with highest Click-through rate""",
    )
    return top_target_ctr,


@app.cell
def __(mo):
    mo.md(
        r"""
        -------------------------------------
        ### Campaigns and Target Groups with **Lowest CPC**
        """
    )
    return


@app.cell
def __(mo):
    bottom_cpc = mo.ui.slider(1, 30, label=f"Lowest CPC")
    bottom_cpc
    return bottom_cpc,


@app.cell
def __(bottom_cpc, gp_target_properties, mo):
    top_target_cpc = gp_target_properties.sort_values('cpc', ascending=True).head(bottom_cpc.value).reset_index()[['xyz_campaign_id', 'age', 'interest', 'cpc']]
    mo.ui.table(
        data=top_target_cpc,
        # use pagination when your table has many rows
        pagination=False,
        label=f"""Lowest {bottom_cpc.value} campaigns and target groups (age & interest) with lowest Cost Per Click""",
    )
    return top_target_cpc,


@app.cell
def __(mo):
    mo.md(
        r"""
        -------------------------------------
        ### Campaigns and Target Groups with **Highest Approved Conversion Rate**
        """
    )
    return


@app.cell
def __(mo):
    top_cr = mo.ui.slider(1, 50, label=f"Top Conversion Rate")
    top_cr
    return top_cr,


@app.cell
def __(gp_target_properties, mo, top_cr):
    top_target_cr = gp_target_properties.sort_values('approved_cr', ascending=False).head(top_cr.value).reset_index()[['xyz_campaign_id', 'age', 'interest', 'Clicks', 'approved_cr']]
    mo.ui.table(
        data=top_target_cr,
        # use pagination when your table has many rows
        pagination=False,
        label=f"""Top {top_cr.value} campaigns and target groups (age & interest) with highest Approved Conversion Rate""",
    )
    return top_target_cr,


@app.cell
def __(mo):
    mo.md(r"""> **Conversion Rates above 100% reveal View-Through Conversions**""")
    return


@app.cell
def __(mo):
    mo.md(r"""### Campaigns and Target Groups with **Lowest Cost per Conversion**""")
    return


@app.cell
def __(mo):
    bottom_cost_per_conversion = mo.ui.slider(1, 50, label=f"Lowest Cost per Conversion")
    bottom_cost_per_conversion
    return bottom_cost_per_conversion,


@app.cell
def __(bottom_cost_per_conversion, gp_target_properties, mo):
    top_target_cost_per_conversion = gp_target_properties.sort_values('cost_per_conversion', ascending=True).head(bottom_cost_per_conversion.value).reset_index()[['xyz_campaign_id', 'age', 'interest', 'Clicks', 'Approved_Conversion','cost_per_conversion']]
    mo.ui.table(
        data=top_target_cost_per_conversion,
        # use pagination when your table has many rows
        pagination=False,
        label=f"""Lowest {bottom_cost_per_conversion.value} campaigns and target groups (age & interest) with lowest Cost per Conversion""",
    )
    return top_target_cost_per_conversion,


@app.cell
def __(mo):
    mo.md(
        r"""
        -------------------------------------
        ### Example Scenario (Target-based)
        """
    )
    return


@app.cell
def __(np, pd):
    def conversion_target(
        campaigns: list, 
        ages: list, 
        interests: list, 
        budgets: list, 
        campaign_properties: pd.DataFrame) -> dict:
        """
        return dict
        """
        clicks_list = []
        impressions_list = []
        conversions_list = []
        campaign_properties = campaign_properties.set_index(['xyz_campaign_id', 'age', 'interest'])
        for campaign, age, interest, budget in zip(campaigns, ages, interests, budgets):
            CTR = campaign_properties.loc[(int(campaign), age, int(interest))]['ctr']
            CPC = campaign_properties.loc[(int(campaign), age, int(interest))]['cpc']
            ACR = campaign_properties.loc[(int(campaign), age, int(interest))]['approved_cr'] # Approved conversion rate
            # print('------------------------------')
            # print(campaign, age, interest, budget)
            # print(f"CPC: {CPC}, CTR: {CTR}, ACR: {ACR}")
            clicks = int(np.round(budget * (1/(CPC))))
            impressions = int(np.round(clicks * (100/CTR)))
            conversions = int(np.round(clicks * ACR/100))
            clicks_list.append(clicks)
            impressions_list.append(impressions)
            conversions_list.append(conversions)
        return dict(impressions = impressions_list, clicks = clicks_list, conversions = conversions_list)
    return conversion_target,


@app.cell
def __(gp_target_properties):
    gp_target_properties_grid = gp_target_properties.reset_index()
    return gp_target_properties_grid,


@app.cell
def __(center_bold_text):
    # mo.md(r"""**Select number of campaign and target groups**""")
    center_bold_text("Select number of campaign and target groups")
    return


@app.cell
def __(mo):
    scenario_size = mo.ui.slider(1, 50, label=f"Scenario Elements")
    scenario_size
    return scenario_size,


@app.cell
def __(gp_target_properties_grid, mo, scenario_size):
    # Campaigns dropdown
    campaign_ids = [str(x) for x in gp_target_properties_grid.xyz_campaign_id]
    campaign_drop_single = mo.ui.dropdown(
            campaign_ids,
            value=campaign_ids[0],
            label="",
        )        

    campaign_drop_multiple = mo.ui.array([campaign_drop_single] * scenario_size.value, label="Campaigns")
    return campaign_drop_multiple, campaign_drop_single, campaign_ids


@app.cell
def __(campaign_drop_multiple, gp_target_properties_grid, mo):
    # Age dropdown
    age_drop_list = []
    for campaign_i in campaign_drop_multiple.value:
        age_categories_i = list(gp_target_properties_grid[gp_target_properties_grid.xyz_campaign_id==int(campaign_i)]['age'])
        age_drop_i = mo.ui.dropdown(
            age_categories_i,
            value=age_categories_i[0],
            label=""
        )
        age_drop_list.append(age_drop_i)

    age_drop_multiple = mo.ui.array(age_drop_list, label=f"""Ages""")
    return (
        age_categories_i,
        age_drop_i,
        age_drop_list,
        age_drop_multiple,
        campaign_i,
    )


@app.cell
def __(
    age_drop_multiple,
    campaign_drop_multiple,
    gp_target_properties_grid,
    mo,
):
    # Interest dropdown
    interest_drop_list = []
    for campaign_ii, age_ii in zip(campaign_drop_multiple.value, age_drop_multiple.value):
        interest_categories = list(
            gp_target_properties_grid[
            (gp_target_properties_grid.xyz_campaign_id==int(campaign_ii)) & (gp_target_properties_grid.age==age_ii)
            ]['interest']
        )
        interest_categories = [str(x) for x in interest_categories]
        interest_drop_i = mo.ui.dropdown(
            interest_categories,
            value=interest_categories[0],
            label=""
        )
        interest_drop_list.append(interest_drop_i)

    interest_drop_multiple = mo.ui.array(interest_drop_list, label="Interests")
    return (
        age_ii,
        campaign_ii,
        interest_categories,
        interest_drop_i,
        interest_drop_list,
        interest_drop_multiple,
    )


@app.cell
def __(gp_target_properties_grid, mo, scenario_size):
    # Budget input number
    number_single = mo.ui.number(start=0.0, stop=gp_target_properties_grid.Spent.sum(), label="")
    budget_number_multiple = mo.ui.array([number_single] * scenario_size.value, label="Budgets")
    return budget_number_multiple, number_single


@app.cell
def __(mo):
    mo.md(r"""----------------------""")
    return


@app.cell
def __(center_bold_text):
    # mo.md(
    #     r"""
    #     -------------------------------------
    #     **Visualize campaign and target groups by performance (e.g. `ctr`, `cpc`, `approved_cr`)**
    #     """
    # )
    center_bold_text("Visualize campaign and target groups by performance (e.g. `ctr`, `cpc`, `approved_cr`)")
    return


@app.cell
def __(gp_target_properties, mo):
    mo.ui.dataframe(gp_target_properties)
    return


@app.cell
def __(mo):
    mo.md(r"""------------------------------""")
    return


@app.cell
def __(center_bold_text):
    center_bold_text("Populate What-If Scenario")
    return


@app.cell
def __(
    age_drop_multiple,
    budget_number_multiple,
    campaign_drop_multiple,
    interest_drop_multiple,
    mo,
):
    mo.hstack(
        [campaign_drop_multiple, age_drop_multiple, interest_drop_multiple, budget_number_multiple], 
        justify="start", 
        gap=True, 
        widths="equal"
    )
    return


@app.cell
def __(
    age_drop_multiple,
    budget_number_multiple,
    campaign_drop_multiple,
    conversion_target,
    gp_target_properties_grid,
    interest_drop_multiple,
):
    scenario_target = conversion_target(campaign_drop_multiple.value, age_drop_multiple.value, interest_drop_multiple.value, budget_number_multiple.value, gp_target_properties_grid)
    return scenario_target,


@app.cell
def __(center_bold_text):
    # mo.md(r"""**What-If Scenario output**""")
    center_bold_text("What-If Scenario output")
    return


@app.cell
def __(
    age_drop_multiple,
    budget_number_multiple,
    campaign_drop_multiple,
    interest_drop_multiple,
    mo,
    np,
    scenario_target,
):
    def print_bold(text):
        return mo.md(f"**{text}**")

    mo.ui.table(
            [
                {"Campaigns": c_i, "Ages": a_i, "Interests": i_i, "Budgets": b_i, 
                 "Impressions": imp_i, "Clicks": cl_i, "Conversions": conv_i} 
                for c_i, a_i, i_i, b_i, imp_i, cl_i, conv_i in zip(
                    campaign_drop_multiple.value, 
                    age_drop_multiple.value, 
                    interest_drop_multiple.value, 
                    budget_number_multiple.value, 
                    scenario_target['impressions'], 
                    scenario_target['clicks'], 
                    scenario_target['conversions'])
            ] + [
                {"Campaigns": "", "Ages": "", "Interests": "", "Budgets": "", 
                 "Impressions": "", "Clicks": "",
                  "Conversions": ""} 
            ] + [
                {"Campaigns": "", "Ages": "", "Interests": "Total: ", "Budgets": np.sum(budget_number_multiple.value), 
                 "Impressions": np.sum(scenario_target['impressions']), "Clicks": np.sum(scenario_target['clicks']),
                  "Conversions": np.sum(scenario_target['conversions'])} 
            ],
            selection=None,
            format_mapping={
                "Impressions": print_bold,  # Use callable to format first names
                "Clicks": print_bold,  # Use string format for age
                "Conversions": print_bold
            }
        )
    return print_bold,


@app.cell
def __():
    # tmp = wishes.value + ['hello']
    # tmp
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        # Additional EDA on segment  

        * Segments are defined by age, gender, and interest.
        * Because segments can respond differently to different campaigns, we can add the campaign to segments
        * It will also be of value to see how adds performance varies within segments accross campaigns (xyz_campaign_id)
        """
    )
    return


@app.cell
def __(processed_data):
    processed_data_segments = (
        processed_data
        .assign(
            segments = processed_data.apply(lambda x: '_'.join([x['age'], x['gender'], str(x['interest'])]), axis=1),
            ctr = lambda x: 100*(x['Clicks']/x['Impressions']),
            cpc = lambda x: (x['Spent']/x['Clicks']),
            total_cr = lambda x: 100*x['Total_Conversion']/x['Clicks'],
            approved_cr = lambda x: 100*x['Approved_Conversion']/x['Clicks'],
            cost_per_conversion = lambda x: x['Spent']/x['Approved_Conversion']
        )
    )

    processed_data_segments = processed_data_segments[processed_data_segments.Clicks>=processed_data_segments.Total_Conversion]

    return processed_data_segments,


@app.cell
def __(plt, processed_data_segments, sns):
    plt.figure(figsize=(18,18))
    sns.pairplot(data=processed_data_segments[['age', 'gender','interest', 'Impressions', 'Clicks', 'Spent', 'Total_Conversion', 'Approved_Conversion', 'segments']], hue="segments", plot_kws={'alpha': 0.5}, palette='Set2') # ,hue="Product"
    plt.legend(loc = 'lower left')
    plt.show();

    return


@app.cell
def __(plt, processed_data_segments, sns):
    plt.figure(figsize=(6,6))
    g = sns.pairplot(data=processed_data_segments[['segments', 'ctr', 'total_cr', 'approved_cr']], hue="segments", plot_kws={'alpha': 0.5, 'legend': False}, palette='Set2') # ,hue="Product"
    # Remove legend
    g._legend.remove()
    plt.show();

    return g,


@app.cell
def __(plt, processed_data_segments, sns):
    # plt.figure(figsize=(40,2))
    sns.displot(data=processed_data_segments, x="ctr", kind="kde", hue='segments', alpha= 0.5, legend=False, palette='Set2', aspect=2) #kind="kde"
    plt.xlim(0,0.1)
    plt.title("Conversion Click-Through Rate distribution per segment")
    plt.show();

    return


@app.cell
def __(plt, processed_data_segments, sns):
    plt.figure(figsize=(40,2))
    sns.displot(data=processed_data_segments, x="total_cr", kde=True, hue='segments', alpha= 0.5, legend=False, palette='Set2', aspect=2)
    plt.xlim(-5,110)
    plt.title("Conversion Rate Distribution per segment")
    plt.show();

    return


@app.cell
def __(plt, processed_data_segments, sns):
    plt.figure(figsize=(3,48))
    sns.barplot(data=processed_data_segments,
                x = "ctr",
                y = "segments",
                hue= "xyz_campaign_id",
                errorbar=None,
                palette='Set2'
               )
    # plt.legend(bbox_to_anchor=(1.00, 1))
    plt.show()

    # [(processed_data_segments.xyz_campaign_id==936)|(processed_data_segments.xyz_campaign_id==916)]
    return


if __name__ == "__main__":
    app.run()
