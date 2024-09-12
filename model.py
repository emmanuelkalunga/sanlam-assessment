import marimo

__generated_with = "0.8.2"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from enum import Enum
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
    import pymc as pm
    import arviz as az
    import scipy.stats as stats
    return (
        Enum,
        az,
        mean_absolute_error,
        mo,
        np,
        pd,
        plt,
        pm,
        r2_score,
        root_mean_squared_error,
        sns,
        stats,
        train_test_split,
    )


@app.cell
def __():
    import IPython.display as display
    return display,


@app.cell
def __(mo):
    mo.md(
        r"""
        #Predictive modelling
        We will model the dependency between variables and propose a simple bayasian model.
        """
    )
    return


@app.cell
def __(pd):
    raw_data = pd.read_csv('conversion_data.csv')
    return raw_data,


@app.cell
def __(raw_data):
    raw_data.columns
    return


@app.cell
def __(raw_data):
    # Remove view-through conversions
    raw_data_filtered = raw_data[raw_data.Clicks>=raw_data.Total_Conversion]
    return raw_data_filtered,


@app.cell
def __(Enum):
    class Campaign(Enum):
        COMP1 = 916
        COMP2 = 936
        COMP3 = 1178
    return Campaign,


@app.cell
def __(mo):
    def center_bold_text(text):
        return mo.md(f"<div style='text-align: center;'><b>{text}</div>")
    return center_bold_text,


@app.cell
def __(Campaign, raw_data_filtered):
    raw_data_camp1 = raw_data_filtered[raw_data_filtered.xyz_campaign_id==Campaign.COMP1.value]
    raw_data_camp2 = raw_data_filtered[raw_data_filtered.xyz_campaign_id==Campaign.COMP2.value]
    raw_data_camp3 = raw_data_filtered[raw_data_filtered.xyz_campaign_id==Campaign.COMP3.value]
    return raw_data_camp1, raw_data_camp2, raw_data_camp3


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Splitting data. 

        Splitting data into train and test, ensuring that target groups (or segments) are equally represented on both ends
        """
    )
    return


@app.cell
def __(raw_data_filtered):
    raw_data_filtered.columns
    return


@app.cell
def __(raw_data_filtered):
    raw_data_filtered_segments = (
        raw_data_filtered
        .assign(
            segments = raw_data_filtered.apply(lambda x: '_'.join([str(x['xyz_campaign_id']), x['age'], x['gender']]), axis=1),
            targets = raw_data_filtered.apply(lambda x: '_'.join([x['age'], x['gender']]), axis=1)
        )
    )
    #str(x['interest'])
    # [str(x['xyz_campaign_id']), x['age'], x['gender']]
    return raw_data_filtered_segments,


@app.cell
def __(raw_data_filtered_segments):
    raw_data_filtered_segments.segments.value_counts()[-10:]
    return


@app.cell
def __(raw_data_filtered_segments, train_test_split):
    # Split the data into train and test sets (80% train, 10% test)

    train_data, test_data = train_test_split(
        raw_data_filtered_segments, 
        test_size=0.2, 
        stratify=raw_data_filtered_segments['segments'],
        random_state=42) 

    print("Training Data:")
    print(train_data.shape)
    print("\nTesting Data:")
    print(test_data.shape)
    return test_data, train_data


@app.cell
def __(mo):
    mo.md(
        """
        # Simple model - Pooled model

        Let $I$, $C$, $S$, $Y$, and $\ddot{Y}$ be respectively impressions, clicks, spend, conversions, and approved conversion.   Then,

        \[\ddot{Y} = f(Y)\]

        \[Y = f(C)\]

        \[C = f(I)\]

        \[S = f(C)\]
        """
    )
    return


@app.cell
def __(train_data):
    # data = train_data.copy()
    data = train_data.copy()
    data = data.assign(
        ad_id = lambda x: x["ad_id"].astype("category"),
        xyz_campaign_id = lambda x: x["xyz_campaign_id"].astype("category"),
        fb_campaign_id = lambda x: x["fb_campaign_id"].astype("category"),
        age = lambda x: x["age"].astype("category"),
        gender = lambda x: x["gender"].astype("category"),
        interest = lambda x: x["interest"].astype("category")    
    )
    return data,


@app.cell
def __(data, np, plt, sns):
    g = sns.pairplot(data=data.drop(["ad_id", "fb_campaign_id"],axis=1)) # ,hue="Product"
    for i, j in zip(*np.triu_indices_from(g.axes, 1)):
        g.axes[i, j].set_visible(False)

    plt.show()
    return g, i, j


@app.cell
def __(mo):
    mo.md(r"""## Relevant Segments""")
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        * From the EDA, it appears that the interest does not contribute much to the click-through rate, nor to the conversion rate. For this reason, in our first iteration, we will only include `age` and `gender` in our segment definition.
        * To take into account all campaigns, we will include campaign into the decisions outcome, making it a "pseudo segment"
        """
    )
    return


@app.cell
def __(data, mo):
    mo.md(
        f"""
        **There is a total of {data.targets.nunique()} targets, and {data.segments.nunique()} possible decision outcomes**
        """
    )
    return


@app.cell
def __(data, np, plt, sns):
    g2 = sns.pairplot(data=data.drop(["ad_id", "fb_campaign_id"],axis=1), hue='segments', plot_kws={'alpha': 0.5}, palette='bright') # ,hue="Product"
    for i2, j2 in zip(*np.triu_indices_from(g2.axes, 1)):
        g2.axes[i2, j2].set_visible(False)

    plt.show()
    return g2, i2, j2


@app.cell
def __(data, np, plt, sns):
    g3 = sns.pairplot(data=data.drop(["ad_id", "fb_campaign_id"],axis=1), hue='targets', plot_kws={'alpha': 0.5}, palette='bright') # ,hue="Product"
    for i3, j3 in zip(*np.triu_indices_from(g3.axes, 1)):
        g3.axes[i3, j3].set_visible(False)

    plt.show()
    return g3, i3, j3


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Modelling Clicks. 

        \[C = f(I)\]

        Clicks shows a linear relation with impressions

        \[C \sim \mathcal{N}(\mu, \sigma^2)\]

        Where 

        \[\mu =  a \times I + b\]

        * We can first try with a constant $\sigma$ throughout values of $I$, then make it vary with it.  
        * However given the nature of Impressions and Clicks, they are best represented by a Binomial distribution.

        \[C \sim Bin(I, \mu)\]

        Our task will be to find what $\mu$ should be given our observations (ads outcome)
        """
    )
    return


@app.cell
def __(data, plt, sns):
    plt.figure(figsize=(12, 6))
    sns.set_palette("terrain") # nipy_spectral, hot, gnuplot2
    sns.scatterplot(data=data, x='Impressions', y='Clicks')
    plt.show()
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        __Choice of prior__  

        * We will use a non-informative Beta prior: $\mu \sim Beta(a,b)$ where a = b = 1. 
        * This is a flat distribution i.e. $\mu$ can take any value from 0 to 1 with equal probability.
        """
    )
    return


@app.cell
def __(np, plt, stats):
    plt.figure(figsize=(12, 2))
    a = 1
    b = 1
    x = np.linspace(0.01, .99, 100)
    y = stats.beta.pdf(x, a, b)
    lines = plt.plot(x, y, label = f"({a:.1f},{b:.1f})", lw = 3) #  "(%.1f,%.1f)"%(a,b)
    plt.fill_between(x, 0, y, alpha = 0.2, color = lines[0].get_color())
    plt.autoscale(tight=True)
    plt.ylim(0)
    plt.legend(loc = 'upper left', title="(a,b)-parameters")
    plt.title("Non-informative Beta prior");
    return a, b, lines, x, y


@app.cell
def __(data):
    data.columns
    return


@app.cell
def __(data, np, pm):
    idx_labels = np.arange(len(data.Impressions))
    with pm.Model() as simple_model:  #coords={"idx": idx_labels}
        # Data
        simple_model.add_coord("idx", idx_labels, mutable=True)
        n_impressions = pm.MutableData("n_impressions", data.Impressions.values, dims="idx")
        n_clicks = pm.MutableData("n_clicks", data.Clicks.values, dims="idx")
        # prior
        alpha_prior = 1
        beta_prior = 1
        p = pm.Beta('p', alpha=alpha_prior, beta=beta_prior)

        #Likelihood: The probability of the observed data given p
        clicks_number = pm.Binomial('clicks_number', n=n_impressions, p=p, observed=n_clicks, 
                                 dims="idx", shape=n_impressions.shape[0])
        
        # Posterior inference (fitting the model)
        trace = pm.sample(1000, return_inferencedata=True)
    return (
        alpha_prior,
        beta_prior,
        clicks_number,
        idx_labels,
        n_clicks,
        n_impressions,
        p,
        simple_model,
        trace,
    )


@app.cell
def __(pm, simple_model):
    graph = pm.model_to_graphviz(simple_model, save='./graph_simple_model.png') # save='./graph'
    # display.display(graph)
    return graph,


@app.cell
def __(mo):
    mo.image(
        src="./graph_simple_model.png",
        width=350
    )
    return


@app.cell
def __(plt, pm, trace):
    # plt.figure(figsize=(50,3))
    pm.plot_trace(trace)
    plt.show()
    return


@app.cell
def __(pm, trace):
    print(pm.summary(trace, round_to=5))
    return


@app.cell
def __(az, trace):
    az.plot_posterior(trace);
    return


@app.cell
def __(pm, simple_model, trace):
    with simple_model:
        trace.extend(pm.sample_posterior_predictive(trace = trace, extend_inferencedata=True))
    return


@app.cell
def __(az, data, plt, trace):
    fig, ax = plt.subplots()
    az.plot_ppc(trace, ax=ax)
    ax.axvline(data.Clicks.mean(), ls="--", color="r", label="True mean")
    ax.legend(fontsize=10);
    return ax, fig


@app.cell
def __(trace):
    trace
    return


@app.cell
def __(az, data, plt, trace):
    plt.figure(figsize=(12,6))
    ax2 = az.plot_hdi(
        data.Impressions,
        trace.posterior_predictive["clicks_number"],
        hdi_prob=0.95,
        fill_kwargs={"color": "tab:orange", "alpha": 0.9},
    )
    ax2.plot(
        data.Impressions,
        trace.posterior_predictive["clicks_number"].mean(("chain", "draw")),
        label="Posterior predictive mean",
    )
    ax2 = az.plot_lm(
        idata=trace,
        y="clicks_number",
        x="n_impressions",
        kind_pp="hdi",
        y_kwargs={"color": "k", "ms": 6, "alpha": 0.30},
        y_hat_fill_kwargs=dict(fill_kwargs={"color": "tab:orange", "alpha": 0.5 }, hdi_prob=0.99999),
        axes=ax2,
    )
    plt.show()
    return ax2,


@app.cell
def __(mo):
    mo.md(r"""### Predict""")
    return


@app.cell
def __(pm, simple_model, trace):
    # Test
    with simple_model:
        pm.set_data(
            {
                "n_impressions": [2000000], 
                "n_clicks": [0]
            },
            coords={"idx": [0]}
        ) 
        #"n_impressions": new_impressions, "n_clicks": [0]
        test_ppc = pm.sample_posterior_predictive(trace = trace, predictions=True)
    return test_ppc,


@app.cell
def __(test_ppc):
    test_ppc
    return


@app.cell
def __(test_ppc):
    test_ppc.predictions["clicks_number"].mean(dim=["draw", "chain"]).values
    return


@app.cell
def __(az, test_ppc):
    az.plot_posterior(test_ppc, group="predictions");
    return


@app.cell
def __(mo):
    mo.md(r"""### Simple Model Performance""")
    return


@app.cell
def __(az, pm, simple_model, trace):
    with simple_model:
        pm.compute_log_likelihood(trace)

    simple_model_loo = az.loo(trace)
    return simple_model_loo,


@app.cell
def __(simple_model_loo):
    simple_model_loo
    return


@app.cell
def __(trace):
    trace
    return


@app.cell
def __(np, pm, simple_model, test_data, trace):
    # Test
    X_i = test_data.Impressions.values
    Y_i = test_data.Clicks.values
    with simple_model:
        pm.set_data(
            {
                "n_impressions": X_i, 
                "n_clicks": Y_i
            },
            coords={"idx": np.arange(len(X_i))}
        ) 
        #"n_impressions": new_impressions, "n_clicks": [0]
        # test_i_ppc = pm.sample_posterior_predictive(trace = trace, predictions=True)
        test_i_ppc = pm.sample_posterior_predictive(trace = trace)
    return X_i, Y_i, test_i_ppc


@app.cell
def __(test_i_ppc):
    # test_i_ppc.predictions["clicks_number"].mean(dim=["draw", "chain"]).values
    test_i_ppc.posterior_predictive["clicks_number"].mean(dim=["draw", "chain"]).values
    return


@app.cell
def __(test_i_ppc):
    test_i_ppc
    return


@app.cell
def __():
    return


@app.cell
def __(az, plt, test_i_ppc):
    # plt.figure(figsize=(12,6))
    az.plot_ppc(test_i_ppc, figsize=(12,6))
    plt.title("Test Data Posterior Predictive (Number of Clicks")
    plt.show()
    return


@app.cell
def __(az, plt, sns, test_data, test_i_ppc):
    plt.figure(figsize=(12,6))
    ax3 = az.plot_hdi(
        test_data.Impressions,
        test_i_ppc.posterior_predictive["clicks_number"],
        hdi_prob=0.95,
        fill_kwargs={"color": "tab:orange", "alpha": 0.9},
    )
    ax3.plot(
        test_data.Impressions,
        test_i_ppc.posterior_predictive["clicks_number"].mean(("chain", "draw")),
        label="Posterior predictive mean",
    )
    ax3 = az.plot_lm(
        idata=test_i_ppc,
        y="clicks_number",
        x="n_impressions",
        kind_pp="hdi",
        y_kwargs={"color": "k", "ms": 6, "alpha": 0.30},
        y_hat_fill_kwargs=dict(fill_kwargs={"color": "tab:orange", "alpha": 0.5 }, hdi_prob=0.99999),
        axes=ax3,
    )
    sns.scatterplot(x=test_data.Impressions.values, 
                    y=test_i_ppc.posterior_predictive["clicks_number"].mean(dim=["draw", "chain"]).values)
    plt.title("Test Data Impressions versusus Clicks")
    plt.show()
    return ax3,


@app.cell
def __(
    mean_absolute_error,
    mo,
    r2_score,
    root_mean_squared_error,
    test_data,
    test_i_ppc,
):
    y_pred = test_i_ppc.posterior_predictive["clicks_number"].mean(dim=["draw", "chain"]).values
    y_true = test_data.Clicks.values
    # RMSE
    rmse = root_mean_squared_error(y_true, y_pred)  # Set squared=False to get RMSE
    # MAE
    mae = mean_absolute_error(y_true, y_pred)
    # R-squared
    r2 = r2_score(y_true, y_pred)

    mo.md(
        f"""
        * **RMSE**: {rmse:.3f}. 
        * **MAE**: {mae:.3f}
        * **R-squared**: {r2:.3f}
        """
    )
    return mae, r2, rmse, y_pred, y_true


@app.cell
def __(mo):
    mo.md(r"""# Hierarchical approach: campaign""")
    return


@app.cell
def __(data, idx_labels, np, pm):
    # campaign_ids = data.xyz_campaign_id.apply(lambda x: str(x)).values
    campaign_ids = data.xyz_campaign_id.values
    unique_campaigns, campaign_idx = np.unique(campaign_ids, return_inverse=True)
    n_campaigns = len(unique_campaigns)
    # Create a dictionary that maps each unique campaign value to its zero-based index
    campaign_mapping = {campaign: idx for idx, campaign in enumerate(unique_campaigns)}

    with pm.Model() as hierarchical_model_campaign:  #coords={"idx": idx_labels}
        # Coordinates
        hierarchical_model_campaign.add_coord("campaign", unique_campaigns, mutable=True)
        hierarchical_model_campaign.add_coord("idx", idx_labels, mutable=True)
        
        # Data
        n_impressions_hc = pm.MutableData("n_impressions_hc", data.Impressions.values, dims="idx")
        n_clicks_hc = pm.MutableData("n_clicks_hc", data.Clicks.values, dims="idx")
        # campaign = pm.MutableData("campaign", campaign_idx, dims="idx") # Link the campaign to observations
        campaign_hc = pm.MutableData("campaign_hc", campaign_idx, dims="idx") # Link the campaign to observations

        # Hierarchical priors for the campaign-level probabilities
        alpha_prior_hc = pm.HalfNormal("alpha_prior_hc", sigma=2)
        beta_prior_hc = pm.HalfNormal("beta_prior_hc", sigma=2)

        # Campaign-specific probabilities p
        p_campaign = pm.Beta("p_campaign", alpha=alpha_prior_hc, beta=beta_prior_hc, dims="campaign")

        # Use campaign-specific probabilities for each observation
        # p_hc = p_campaign[campaign]
        p_hc = p_campaign[campaign_idx]

        # Likelihood: The probability of the observed data given p
        clicks_number_hc = pm.Binomial('clicks_number_hc', n=n_impressions_hc, p=p_hc, observed=n_clicks_hc, 
                                       dims="idx", shape=n_impressions_hc.shape[0])

        # Posterior inference (fitting the model)
        trace_hc = pm.sample(1000, return_inferencedata=True)
    return (
        alpha_prior_hc,
        beta_prior_hc,
        campaign_hc,
        campaign_ids,
        campaign_idx,
        campaign_mapping,
        clicks_number_hc,
        hierarchical_model_campaign,
        n_campaigns,
        n_clicks_hc,
        n_impressions_hc,
        p_campaign,
        p_hc,
        trace_hc,
        unique_campaigns,
    )


@app.cell
def __(hierarchical_model_campaign, pm):
    graph_hierarchical_model_campaign = pm.model_to_graphviz(
        hierarchical_model_campaign, 
        save='./graph_hierarchical_model_campaign.png') # save='./graph'
    return graph_hierarchical_model_campaign,


@app.cell
def __(mo):
    mo.image(
        src="./graph_hierarchical_model_campaign.png",
        width=600
    )
    return


@app.cell
def __(plt, pm, trace_hc):
    pm.plot_trace(trace_hc, figsize=(18,10), legend=True)
    plt.show()
    return


@app.cell
def __(az, plt, trace_hc):
    az.plot_forest(trace_hc, var_names="p_campaign", figsize=(12,2), combined=True)
    # plt.grid(True)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()
    return


@app.cell
def __(hierarchical_model_campaign, pm, trace_hc):
    with hierarchical_model_campaign:
        trace_hc.extend(pm.sample_posterior_predictive(trace = trace_hc, extend_inferencedata=True))
    return


@app.cell
def __(az, data, plt, sns, trace_hc):
    plt.figure(figsize=(12,6))
    ax5 = az.plot_hdi(
        data.Impressions,
        trace_hc.posterior_predictive["clicks_number_hc"],
        hdi_prob=0.95,
        fill_kwargs={"color": "tab:orange", "alpha": 0.9},
    )
    ax5.plot(
        data.Impressions,
        trace_hc.posterior_predictive["clicks_number_hc"].mean(("chain", "draw")),
        label="Posterior predictive mean",
    )
    ax5 = az.plot_lm(
        idata=trace_hc,
        y="clicks_number_hc",
        x="n_impressions_hc",
        kind_pp="hdi",
        y_kwargs={"color": "k", "ms": 6, "alpha": 0.30},
        y_hat_fill_kwargs=dict(fill_kwargs={"color": "tab:orange", "alpha": 0.5 }, hdi_prob=0.99999),
        axes=ax5,
    )
    sns.scatterplot(x=data.Impressions.values, 
                    y=trace_hc.posterior_predictive["clicks_number_hc"].mean(dim=["draw", "chain"]).values)
    plt.title("Impressions versusus Clicks")
    plt.show()
    return ax5,


@app.cell
def __(az, plt, trace_hc):
    az.plot_posterior(trace_hc)
    # az.plot_posterior(trace_hc, group="predictions");
    plt.show()
    return


@app.cell
def __(az, data, plt, trace_hc):
    fig4, ax4 = plt.subplots(figsize=(12,6))
    az.plot_ppc(trace_hc, ax=ax4)
    ax4.axvline(data.Clicks.mean(), ls="--", color="r", label="True mean")
    ax4.legend(fontsize=10);
    ax4.set_xlim(0, 500)
    plt.show()
    return ax4, fig4


@app.cell
def __(mo):
    mo.md(r"""## Predict""")
    return


@app.cell
def __(np):
    np.atleast_1d([1, 2])
    return


@app.cell
def __(campaign_mapping):
    campaign_mapping[1178]
    return


@app.cell
def __(np):
    def predict_clicks_number(trace, campaign_idx, new_n_impressions_hc, target='campaign'):
        """
        Predict the number of clicks given a new value of n_impressions_hc and a specific campaign.
        
        Parameters:
        - trace: The posterior trace from PyMC sampling.
        - campaign_idx: The index of the campaign for which we want to predict clicks.
        - new_n_impressions_hc: The new number of impressions (scalar or array) for which we want to predict clicks.
        
        Returns:
        - Predicted number of clicks (posterior predictive samples).
        """
        # Extract the posterior samples for p_campaign for the specific campaign
        if target == 'campaign':
            p_posterior = trace.posterior["p_campaign"].sel(campaign=campaign_idx).values
        elif target == 'segment':
            p_posterior = trace.posterior["p_segment"].sel(segment=campaign_idx).values

        # Ensure that new_n_impressions_hc is an array
        new_n_impressions_hc = np.atleast_1d(new_n_impressions_hc)
        
        # Use the posterior samples of p_campaign to predict clicks using the Binomial distribution
        predicted_clicks = np.random.binomial(n=new_n_impressions_hc, p=p_posterior)
        
        return predicted_clicks
    return predict_clicks_number,


@app.cell
def __(np, predict_clicks_number):
    def predict_clicks_number_df(trace, frame, 
                                 campaign_col='xyz_campaign_id', 
                                 impressions_col='Impressions', 
                                 clicks_col='Clicks',
                                 campaign_mapping=None, target='campaign'):
        """
        Predict clicks for each row in the DataFrame using the posterior samples of p_campaign.
        
        Parameters:
        - trace: The posterior trace from PyMC sampling.
        - frame: The DataFrame containing campaign data.
        - campaign_col: The name of the column containing campaign IDs.
        - impressions_col: The name of the column containing impressions data.
        - clicks_col: The name of the column containing the observed clicks (optional).
        - campaign_mapping: A dictionary that maps campaign IDs to their zero-based index (optional).
        
        Returns:
        - df: The DataFrame with an additional column 'predicted_clicks' containing the predicted values.
        """
        df = frame[[campaign_col, impressions_col, clicks_col]].copy()
        # df = frame.copy()
        # Initialize a list to store predicted clicks
        predicted_clicks_list = []
        predicted_clicks_mean_list = []
        predicted_clicks_ci_95_list = []
        predicted_clicks_ci_95_lower_list = []
        predicted_clicks_ci_95_upper_list = []

        # # Ensure the campaign mapping exists
        # if campaign_mapping is None:
        #     campaign_ids = df[campaign_col].values
        #     unique_campaigns, campaign_idx = np.unique(campaign_ids, return_inverse=True)
        #     campaign_mapping = {campaign: idx for idx, campaign in enumerate(unique_campaigns)}
        
        # Iterate over the rows of the DataFrame
        for index, row in df.iterrows():
            # Get the campaign index using the mapping
            # campaign_id = row[campaign_col]
            # campaign_idx = campaign_mapping.get(campaign_id)
            campaign_idx = row[campaign_col]
            
            # Get the number of impressions
            new_n_impressions_hc = row[impressions_col]
            
            # Predict clicks for the current row
            if campaign_idx is not None:
                predicted_clicks = predict_clicks_number(trace, campaign_idx, new_n_impressions_hc, target=target)
                # Store the mean of the predicted clicks for the row
                predicted_clicks_list.append(predicted_clicks)
                predicted_clicks_mean_list.append(predicted_clicks.mean())
                ci = np.percentile(predicted_clicks, [2.5, 97.5])
                predicted_clicks_ci_95_lower_list.append(ci[0])
                predicted_clicks_ci_95_upper_list.append(ci[1])
            else:
                predicted_clicks_list.append(np.nan)  # If no campaign_idx is found, set NaN
                predicted_clicks_mean_list.append(np.nan)
                predicted_clicks_ci_95_list.append(np.nan)

        # Add the predicted clicks as a new column in the DataFrame
        # df['predicted_clicks'] = predicted_clicks_list
        df['predicted_clicks_mean'] = predicted_clicks_mean_list
        df['predicted_clicks_ci_95_lower'] = predicted_clicks_ci_95_lower_list
        df['predicted_clicks_ci_95_upper'] = predicted_clicks_ci_95_upper_list

        return df
    return predict_clicks_number_df,


@app.cell
def __(predict_clicks_number_df, test_data, trace_hc):
    predicted_df = predict_clicks_number_df(trace_hc, test_data)
    return predicted_df,


@app.cell
def __(plt, predicted_df, sns):
    sns.set_palette('bright')
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='Impressions', y='Clicks', data=predicted_df, label="Actual Clicks",  alpha=0.7)

    # Plot the predicted clicks (mean) as a line
    sns.lineplot(x='Impressions', y='predicted_clicks_mean', data=predicted_df, label="Predicted Clicks (Mean)", color='red')

    # Adding the confidence interval as shaded areas
    # Group by Impressions and get the min lower bound and max upper bound for each group
    grouped_df = predicted_df.groupby('Impressions').agg({
        'predicted_clicks_ci_95_lower': 'min',
        'predicted_clicks_ci_95_upper': 'max',
        'predicted_clicks_mean': 'mean',
        'Clicks': 'mean'
    }).reset_index()
    plt.fill_between(grouped_df['Impressions'], grouped_df['predicted_clicks_ci_95_lower'], 
                     grouped_df['predicted_clicks_ci_95_upper'], 
                     color='gray', alpha=0.3, label="95% Confidence Interval")


    # plt.xlim(0, 500000)
    # Labels and title
    plt.xlabel('Impressions')
    plt.ylabel('Clicks')
    plt.title('Impressions vs Clicks with Predictions and Confidence Interval')
    plt.legend()

    # Show the plot
    plt.show()
    return grouped_df,


@app.cell
def __():
    return


@app.cell
def __(mo):
    mo.md(r"""## Performance""")
    return


@app.cell
def __(predicted_df):
    predicted_df.columns
    return


@app.cell
def __(
    mean_absolute_error,
    mo,
    predicted_df,
    r2_score,
    root_mean_squared_error,
):
    y_pred_hc = predicted_df['predicted_clicks_mean']
    y_true_hc = predicted_df['Clicks']
    # RMSE
    rmse_hc = root_mean_squared_error(y_true_hc, y_pred_hc)  # Set squared=False to get RMSE
    # MAE
    mae_hc = mean_absolute_error(y_true_hc, y_pred_hc)
    # R-squared
    r2_hc = r2_score(y_true_hc, y_pred_hc)

    mo.md(
        f"""
        Hierarchical (campaings) results: 
        
        * **RMSE**: {rmse_hc:.3f}. 
        * **MAE**: {mae_hc:.3f}
        * **R-squared**: {r2_hc:.3f}
        """
    )
    return mae_hc, r2_hc, rmse_hc, y_pred_hc, y_true_hc


@app.cell
def __(mae, mae_hc, pd, r2, r2_hc, rmse, rmse_hc):
    results_1 = {
        'Method': ['Simple (Pooled)', 'Campaign unpooled'],
        'RMSE': [rmse, rmse_hc],
        'MAE': [mae, mae_hc],
        'R2': [r2, r2_hc]
    }
    results_1 = pd.DataFrame(results_1)
    results_1
    return results_1,


@app.cell
def __(plt, pm, trace, trace_hc):
    pm.plot_trace(trace, var_names=['p'])
    pm.plot_trace(trace_hc, var_names=['p_campaign'])
    plt.show()
    return


@app.cell
def __(np, plt, sns, trace, trace_hc, unique_campaigns):

    # Assuming 'trace_hc' is your trace from PyMC sampling
    # Extract posterior samples for 'p_campaign' (or any other variable)
    # This gives you a 3D array: (chains, draws, dimensions)
    p_campaign_samples = trace_hc.posterior['p_campaign'].values

    # Get the number of dimensions
    n_dims = p_campaign_samples.shape[-1]
    n_draws = p_campaign_samples.shape[0]

    for dim in range(n_dims):
        sns.kdeplot(p_campaign_samples[:, :, dim].flatten(), fill=False, label=f'{unique_campaigns[dim]}')  


    p_samples = np.atleast_3d(trace.posterior['p'].values)

    # Get the number of dimensions
    n_dims = p_samples.shape[-1]
    n_draws = p_samples.shape[0]

    for dim in range(n_dims):
        sns.kdeplot(p_samples[:, :, dim].flatten(), fill=False, label=f'pooled')  
    plt.title("p posterior: simple (pooled) versus hierarchical")
    plt.legend()
    plt.show()
        
    return dim, n_dims, n_draws, p_campaign_samples, p_samples


@app.cell
def __(mo):
    mo.md(r"""# Hierarchical: campaign-age-gender""")
    return


@app.cell
def __(data, idx_labels, np, pm):
    segment_ids = data.segments.values
    unique_segments, segment_idx = np.unique(segment_ids, return_inverse=True)
    n_segments = len(unique_segments)
    # Create a dictionary that maps each unique segment value to its zero-based index
    segment_mapping = {segment: idx for idx, segment in enumerate(unique_segments)}

    with pm.Model() as hierarchical_model_segment:  #coords={"idx": idx_labels}
        # Coordinates
        hierarchical_model_segment.add_coord("segment", unique_segments, mutable=True)
        hierarchical_model_segment.add_coord("idx", idx_labels, mutable=True)
        
        # Data
        n_impressions_hs = pm.MutableData("n_impressions_hs", data.Impressions.values, dims="idx")
        n_clicks_hs = pm.MutableData("n_clicks_hs", data.Clicks.values, dims="idx")
        
        segment_hs = pm.MutableData("segment_hs", segment_idx, dims="idx") # Link the segment to observations

        # Hierarchical priors for the segment-level probabilities
        alpha_prior_hs = pm.HalfNormal("alpha_prior_hs", sigma=2)
        beta_prior_hs = pm.HalfNormal("beta_prior_hs", sigma=2)

        # Campaign-specific probabilities p
        p_segment = pm.Beta("p_segment", alpha=alpha_prior_hs, beta=beta_prior_hs, dims="segment")

        # Use segment-specific probabilities for each observation
        p_hs = p_segment[segment_idx]

        # Likelihood: The probability of the observed data given p
        clicks_number_hs = pm.Binomial('clicks_number_hs', n=n_impressions_hs, p=p_hs, observed=n_clicks_hs, 
                                       dims="idx", shape=n_impressions_hs.shape[0])

        # Posterior inference (fitting the model)
        trace_hs = pm.sample(1000, return_inferencedata=True)
    return (
        alpha_prior_hs,
        beta_prior_hs,
        clicks_number_hs,
        hierarchical_model_segment,
        n_clicks_hs,
        n_impressions_hs,
        n_segments,
        p_hs,
        p_segment,
        segment_hs,
        segment_ids,
        segment_idx,
        segment_mapping,
        trace_hs,
        unique_segments,
    )


@app.cell
def __(plt, pm, trace_hs):
    pm.plot_trace(trace_hs, figsize=(18,10), legend=True)
    plt.show()
    return


@app.cell
def __(az, plt, trace_hs):
    az.plot_forest(trace_hs, var_names="p_segment", figsize=(12,8), combined=True)
    # plt.grid(True) 
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()
    return


@app.cell
def __(hierarchical_model_segment, pm, trace_hs):
    with hierarchical_model_segment:
        trace_hs.extend(pm.sample_posterior_predictive(trace = trace_hs, extend_inferencedata=True))
    return


@app.cell
def __(az, data, plt, sns, trace_hs):
    plt.figure(figsize=(12,6))
    ax6 = az.plot_hdi(
        data.Impressions,
        trace_hs.posterior_predictive["clicks_number_hs"],
        hdi_prob=0.95,
        fill_kwargs={"color": "tab:orange", "alpha": 0.9},
    )
    # ax6.plot(
    #     data.Impressions,
    #     trace_hs.posterior_predictive["clicks_number_hs"].mean(("chain", "draw")),
    #     label="Posterior predictive mean",
    # )
    ax6 = az.plot_lm(
        idata=trace_hs,
        y="clicks_number_hs",
        x="n_impressions_hs",
        kind_pp="hdi",
        y_kwargs={"color": "k", "ms": 6, "alpha": 0.30},
        y_hat_fill_kwargs=dict(fill_kwargs={"color": "tab:orange", "alpha": 0.5 }, hdi_prob=0.99999),
        axes=ax6,
    )
    sns.scatterplot(x=data.Impressions.values, 
                    y=trace_hs.posterior_predictive["clicks_number_hs"].mean(dim=["draw", "chain"]).values,
                    label="Posterior predictive mean")
    plt.title("Impressions versusus Clicks (unpooled - segments)")
    plt.show()
    return ax6,


@app.cell
def __(az, data, plt, trace_hs):
    fig7, ax7 = plt.subplots(figsize=(12,6))
    az.plot_ppc(trace_hs, ax=ax7)
    ax7.axvline(data.Clicks.mean(), ls="--", color="r", label="True mean")
    ax7.legend(fontsize=10);
    ax7.set_xlim(0, 500)
    plt.show()
    return ax7, fig7


@app.cell
def __(mo):
    mo.md(r"""## Predict""")
    return


@app.cell
def __(test_data):
    test_data
    return


@app.cell
def __(predict_clicks_number_df, test_data, trace_hs):
    predicted_df_hs = predict_clicks_number_df(trace_hs, test_data, campaign_col='segments', target='segment')
    return predicted_df_hs,


@app.cell
def __(plt, predicted_df_hs, sns):
    sns.set_palette('bright')
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='Impressions', y='Clicks', data=predicted_df_hs, label="Actual Clicks",  alpha=0.7)

    # Plot the predicted clicks (mean) as a line
    sns.lineplot(x='Impressions', y='predicted_clicks_mean', data=predicted_df_hs, label="Predicted Clicks (Mean)", color='red')

    # Adding the confidence interval as shaded areas
    # Group by Impressions and get the min lower bound and max upper bound for each group
    grouped_df_hs = predicted_df_hs.groupby('Impressions').agg({
        'predicted_clicks_ci_95_lower': 'min',
        'predicted_clicks_ci_95_upper': 'max',
        'predicted_clicks_mean': 'mean',
        'Clicks': 'mean'
    }).reset_index()
    plt.fill_between(grouped_df_hs['Impressions'], grouped_df_hs['predicted_clicks_ci_95_lower'], 
                     grouped_df_hs['predicted_clicks_ci_95_upper'], 
                     color='gray', alpha=0.3, label="95% Confidence Interval")


    # plt.xlim(0, 500000)
    # Labels and title
    plt.xlabel('Impressions')
    plt.ylabel('Clicks')
    plt.title('Impressions vs Clicks with Predictions and Confidence Interval')
    plt.legend()

    # Show the plot
    plt.show()
    return grouped_df_hs,


@app.cell
def __(mo):
    mo.md(r"""## Performance""")
    return


@app.cell
def __(
    mean_absolute_error,
    mo,
    predicted_df_hs,
    r2_score,
    root_mean_squared_error,
):
    y_pred_hs = predicted_df_hs['predicted_clicks_mean']
    y_true_hs = predicted_df_hs['Clicks']
    # RMSE
    rmse_hs = root_mean_squared_error(y_true_hs, y_pred_hs)  # Set squared=False to get RMSE
    # MAE
    mae_hs = mean_absolute_error(y_true_hs, y_pred_hs)
    # R-squared
    r2_hs = r2_score(y_true_hs, y_pred_hs)

    mo.md(
        f"""
        Hierarchical (segments) results: 
        
        * **RMSE**: {rmse_hs:.3f}. 
        * **MAE**: {mae_hs:.3f}
        * **R-squared**: {r2_hs:.3f}
        """
    )
    return mae_hs, r2_hs, rmse_hs, y_pred_hs, y_true_hs


@app.cell
def __(mae, mae_hc, mae_hs, pd, r2, r2_hc, r2_hs, rmse, rmse_hc, rmse_hs):
    results_2 = {
        'Method': ['Simple (Pooled)', 'Hierarchies - Campaign', 'Hierarchies - Segment '],
        'RMSE': [rmse, rmse_hc, rmse_hs],
        'MAE': [mae, mae_hc, mae_hs],
        'R2': [r2, r2_hc, r2_hs]
    }
    results_2 = pd.DataFrame(results_2)
    results_2
    return results_2,


@app.cell
def __(pm):
    pm.__version__
    return


@app.cell
def __():
    # model.check_test_point()
    return


@app.cell
def __(mo):
    mo.md(r"""# Including price""")
    return


@app.cell
def __(data):
    data
    return


@app.cell
def __(data, idx_labels, pm, segment_idx, unique_segments):
    with pm.Model() as hierarchical_model_price:  
        # Coordinates
        hierarchical_model_price.add_coord("segment", unique_segments, mutable=True)
        hierarchical_model_price.add_coord("idx", idx_labels, mutable=True)
        
        # Data
        n_impressions_hp = pm.MutableData("n_impressions_hp", data.Impressions.values, dims="idx")
        n_clicks_hp = pm.MutableData("n_clicks_hp", data.Clicks.values, dims="idx")
        price_hp = pm.MutableData("price_hp", data.Spent.values, dims="idx")  # Add price as an observed variable

        segment_hp = pm.MutableData("segment_hp", segment_idx, dims="idx")  # Link the segment to observations

        # Hierarchical priors for the segment-level probabilities
        alpha_prior_hp = pm.HalfNormal("alpha_prior_hp", sigma=2)
        beta_prior_hp = pm.HalfNormal("beta_prior_hp", sigma=2)

        # Segment-specific probabilities p
        p_segment_hp = pm.Beta("p_segment_hp", alpha=alpha_prior_hp, beta=beta_prior_hp, dims="segment")

        # Use segment-specific probabilities for each observation
        p_hp = p_segment_hp[segment_idx]

        # Likelihood for clicks
        clicks_number_hp = pm.Binomial('clicks_number_hp', n=n_impressions_hp, p=p_hp, observed=n_clicks_hp, dims="idx")

        # Add segment-specific parameters for the price model (priors)
        a_segment = pm.Normal("a_segment", mu=0, sigma=10, dims="segment")  # Intercept term
        b_segment = pm.Normal("b_segment", mu=0, sigma=10, dims="segment")  # Slope term

        # Model the price as a linear function of clicks_number_hs
        price_mu_hp = a_segment[segment_idx] + b_segment[segment_idx] * clicks_number_hp

        # Likelihood for the observed price
        sigma_price = pm.HalfNormal("sigma_price", sigma=1)
        price_obs = pm.Normal('price_obs', mu=price_mu_hp, sigma=sigma_price, observed=price_hp, dims="idx")

        # Posterior inference (fitting the model)
        trace_hp = pm.sample(1000, return_inferencedata=True)
    return (
        a_segment,
        alpha_prior_hp,
        b_segment,
        beta_prior_hp,
        clicks_number_hp,
        hierarchical_model_price,
        n_clicks_hp,
        n_impressions_hp,
        p_hp,
        p_segment_hp,
        price_hp,
        price_mu_hp,
        price_obs,
        segment_hp,
        sigma_price,
        trace_hp,
    )


@app.cell
def __(hierarchical_model_price, pm):
    graph_hierarchical_model_price = pm.model_to_graphviz(
        hierarchical_model_price, 
        save='./graph_hierarchical_model_price.png') # save='./graph'
    return graph_hierarchical_model_price,


@app.cell
def __(mo):
    mo.image(
        src="./graph_hierarchical_model_price.png",
        width=700
    )
    return


@app.cell
def __(plt, pm, trace_hp):
    pm.plot_trace(trace_hp, figsize=(18,15), legend=True)
    plt.show()
    return


@app.cell
def __(az, plt, trace_hp):
    az.plot_forest(trace_hp, var_names=['a_segment', 'b_segment'], figsize=(12,10), combined=True)

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()
    return


@app.cell
def __(hierarchical_model_price, pm, trace_hp):
    with hierarchical_model_price:
        trace_hp.extend(pm.sample_posterior_predictive(trace = trace_hp, extend_inferencedata=True))
    return


@app.cell
def __(az, data, plt, sns, trace_hp):
    plt.figure(figsize=(12,6))
    ax8 = az.plot_hdi(
        data.Clicks,
        trace_hp.posterior_predictive["price_obs"],
        hdi_prob=0.95,
        fill_kwargs={"color": "tab:orange", "alpha": 0.9},
    )
    # ax8.plot(
    #     data.Clicks,
    #     trace_hp.posterior_predictive["price_obs"].mean(("chain", "draw")),
    #     label="Posterior predictive mean (Spend)",
    # )
    ax8 = az.plot_lm(
        idata=trace_hp,
        y="price_obs",
        x="n_clicks_hp",
        kind_pp="hdi",
        y_kwargs={"color": "k", "ms": 6, "alpha": 0.30},
        y_hat_fill_kwargs=dict(fill_kwargs={"color": "tab:orange", "alpha": 0.5 }, hdi_prob=0.99999),
        axes=ax8,
    )
    sns.scatterplot(x=data.Clicks.values, 
                    y=trace_hp.posterior_predictive["price_obs"].mean(dim=["draw", "chain"]).values)
    plt.title("Clicks versusus Spend")
    plt.show()
    return ax8,


@app.cell
def __(mo):
    mo.md(r"""# Include Conversion""")
    return


@app.cell
def __(data):
    data
    return


@app.cell
def __(data, idx_labels, pm, segment_idx, unique_segments):
    with pm.Model() as hierarchical_model_conversion:  
        # Coordinates
        hierarchical_model_conversion.add_coord("segment", unique_segments, mutable=True)
        hierarchical_model_conversion.add_coord("idx", idx_labels, mutable=True)
        
        # Data
        n_impressions_hco = pm.MutableData("n_impressions_hco", data.Impressions.values, dims="idx")
        n_clicks_hco = pm.MutableData("n_clicks_hco", data.Clicks.values, dims="idx")
        price_hco = pm.MutableData("price_hco", data.Spent.values, dims="idx")
        conversions_hco = pm.MutableData("conversions_hs", data.Total_Conversion.values, dims="idx")  # Add conversions data

        segment_hco = pm.MutableData("segment_hco", segment_idx, dims="idx")  # Link the segment to observations

        # Hierarchical priors for the segment-level probabilities
        alpha_prior_hco = pm.HalfNormal("alpha_prior_hco", sigma=2)
        beta_prior_hco = pm.HalfNormal("beta_prior_hco", sigma=2)

        # Segment-specific probabilities for clicks
        p_segment_hco = pm.Beta("p_segment_hco", alpha=alpha_prior_hco, beta=beta_prior_hco, dims="segment")

        # Use segment-specific probabilities for each observation
        p_hco = p_segment_hco[segment_idx]

        # Likelihood for clicks
        clicks_number_hco = pm.Binomial('clicks_number_hco', n=n_impressions_hco, p=p_hco, observed=n_clicks_hco, dims="idx")

        # Add segment-specific parameters for the price model
        a_segment_hco = pm.Normal("a_segment_hco", mu=0, sigma=10, dims="segment")  # Intercept term
        b_segment_hco = pm.Normal("b_segment_hco", mu=0, sigma=10, dims="segment")  # Slope term

        # Model the price as a linear function of clicks_number_hs
        price_mu_hco = a_segment_hco[segment_idx] + b_segment_hco[segment_idx] * clicks_number_hco

        # Likelihood for the observed price
        sigma_price_hco = pm.HalfNormal("sigma_price_hco", sigma=1)
        price_obs_hco = pm.Normal('price_obs_hco', mu=price_mu_hco, sigma=sigma_price_hco, observed=price_hco, dims="idx")

        # Conversion model: segment-specific probabilities for conversion
        alpha_conversion = pm.HalfNormal("alpha_conversion", sigma=2)
        beta_conversion = pm.HalfNormal("beta_conversion", sigma=2)

        # Segment-specific conversion probabilities
        p_conversion_segment = pm.Beta("p_conversion_segment", alpha=alpha_conversion, beta=beta_conversion, dims="segment")

        # Use segment-specific probabilities for conversion
        p_conversion_hco = p_conversion_segment[segment_idx]

        # Likelihood for conversions (successes out of clicks_number_hs)
        conversions_obs = pm.Binomial('conversions_obs', n=clicks_number_hco, p=p_conversion_hco, observed=conversions_hco, dims="idx")

        # Posterior inference (fitting the model)
        trace_hco = pm.sample(1000, return_inferencedata=True)
    return (
        a_segment_hco,
        alpha_conversion,
        alpha_prior_hco,
        b_segment_hco,
        beta_conversion,
        beta_prior_hco,
        clicks_number_hco,
        conversions_hco,
        conversions_obs,
        hierarchical_model_conversion,
        n_clicks_hco,
        n_impressions_hco,
        p_conversion_hco,
        p_conversion_segment,
        p_hco,
        p_segment_hco,
        price_hco,
        price_mu_hco,
        price_obs_hco,
        segment_hco,
        sigma_price_hco,
        trace_hco,
    )


@app.cell
def __(hierarchical_model_conversion, pm):
    graph_hierarchical_model_conversion = pm.model_to_graphviz(hierarchical_model_conversion, 
                                                               save='./graph_hierarchical_model_conversion.png')
    return graph_hierarchical_model_conversion,


@app.cell
def __(mo):
    mo.image(
        src="./graph_hierarchical_model_conversion.png",
        width=2000
    )
    return


@app.cell
def __(plt, pm, trace_hco):
    pm.plot_trace(trace_hco, figsize=(18,15), legend=True)
    plt.show()
    return


@app.cell
def __(az, plt, trace_hco):
    az.plot_forest(trace_hco, var_names="p_conversion_segment", figsize=(12,10), combined=True)

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()
    return


@app.cell
def __(hierarchical_model_conversion, pm, trace_hco):
    with hierarchical_model_conversion:
        trace_hco.extend(pm.sample_posterior_predictive(trace = trace_hco, extend_inferencedata=True))
    return


@app.cell
def __(az, data, plt, sns, trace_hco):
    plt.figure(figsize=(12,6))
    ax9 = az.plot_hdi(
        data.Clicks,
        trace_hco.posterior_predictive["conversions_obs"],
        hdi_prob=0.95,
        fill_kwargs={"color": "tab:orange", "alpha": 0.9},
    )
    # ax8.plot(
    #     data.Clicks,
    #     trace_hp.posterior_predictive["price_obs"].mean(("chain", "draw")),
    #     label="Posterior predictive mean (Spend)",
    # )
    ax9 = az.plot_lm(
        idata=trace_hco,
        y="conversions_obs",
        x="n_clicks_hco",
        kind_pp="hdi",
        y_kwargs={"color": "k", "ms": 6, "alpha": 0.30},
        y_hat_fill_kwargs=dict(fill_kwargs={"color": "tab:orange", "alpha": 0.5 }, hdi_prob=0.99999),
        axes=ax9,
    )
    sns.scatterplot(x=data.Clicks.values, 
                    y=trace_hco.posterior_predictive["conversions_obs"].mean(dim=["draw", "chain"]).values,
                    label='Posterior predictive mean (Total Conversion)'
                   )
    plt.title("Clicks versusus Conversion")
    plt.show()
    return ax9,


@app.cell
def __(mo):
    mo.md(r"""## Predict (What-if analysis) - Clicks""")
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        To predict number of clicks from the spend (price), we will invert the linear relationship: 

        \[
        \text{price\_mu} = a + b \times \text{clicks\_number}
        \]

        into 

        \[
        \text{clicks\_number} = \frac{\text{price\_mu} - a}{b}
        \]
        """
    )
    return


@app.cell
def __(np):
    def predict_clicks_given_spend(trace, segment_idx, price_mu_hs):
        """
        Predict the number of clicks (clicks_number_hs) given a price (price_mu_hs) using
        the posterior samples of a_segment and b_segment.
        
        Parameters:
        - trace: The posterior trace from PyMC sampling.
        - segment_idx: The index of the segment for which to predict clicks.
        - price_mu_hs: The observed price for which to predict the number of clicks.
        
        Returns:
        - A NumPy array of predicted clicks based on the posterior samples.
        """
        # Extract posterior samples for a_segment and b_segment for the given segment
        a_samples = trace.posterior['a_segment_hco'].sel(segment=segment_idx).values.flatten()
        b_samples = trace.posterior['b_segment_hco'].sel(segment=segment_idx).values.flatten()
        
        # Ensure b_samples are not zero to avoid division by zero
        b_samples = np.where(b_samples == 0, 1e-6, b_samples)
        
        # Calculate clicks_number_hs for each sample (using the inverted formula)
        predicted_clicks = (price_mu_hs - a_samples) / b_samples
        
        # Ensure the predicted clicks are non-negative (since clicks can't be negative)
        predicted_clicks = np.maximum(predicted_clicks, 0)
        
        return predicted_clicks


    return predict_clicks_given_spend,


@app.cell
def __(np, predict_clicks_given_spend, trace_hco, unique_segments):
    price_mu_hs_test = 1000  # Example price for which we want to predict clicks
    predicted_clicks_mean_list = []
    predicted_clicks_lower_list = []
    predicted_clicks_upper_list = []
    for test_segment_i in unique_segments:
        spend_predicted_clicks_i = predict_clicks_given_spend(trace_hco, test_segment_i, price_mu_hs_test)
        predicted_clicks_mean_list.append(np.mean(spend_predicted_clicks_i))
        ci = np.percentile(spend_predicted_clicks_i, [2.5, 97.5])
        predicted_clicks_lower_list.append(ci[0])
        predicted_clicks_upper_list.append(ci[1])
    return (
        ci,
        predicted_clicks_lower_list,
        predicted_clicks_mean_list,
        predicted_clicks_upper_list,
        price_mu_hs_test,
        spend_predicted_clicks_i,
        test_segment_i,
    )


@app.cell
def __(
    pd,
    predicted_clicks_lower_list,
    predicted_clicks_mean_list,
    predicted_clicks_upper_list,
    unique_segments,
):
    sim_results = {
        "segment": unique_segments,
        "predicted_clicks_mean": predicted_clicks_mean_list,
        "predicted_clicks_lower":predicted_clicks_lower_list,
        "predicted_clicks_upper": predicted_clicks_upper_list,
    }
    sim_results = pd.DataFrame(sim_results)
    return sim_results,


@app.cell
def __(plt, price_mu_hs_test, sim_results, sns):
    plt.figure(figsize=(10, 6))

    # Plot the line for confidence intervals (from lower to upper)
    for row_idx, row in sim_results.iterrows():
        plt.plot([row['predicted_clicks_lower'], row['predicted_clicks_upper']], [row['segment'], row['segment']], color='gray')

    # Plot the dot for the predicted mean
    sns.scatterplot(x='predicted_clicks_mean', y='segment', data=sim_results, color='blue', s=100, label='Predicted Clicks Mean')

    # Add labels and title
    plt.xlabel('Predicted Clicks')
    plt.ylabel('Segment')
    plt.title(f'Predicted Clicks given budget ({price_mu_hs_test}$)  with Confidence Intervals by Segment')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.show()
    return row, row_idx


@app.cell
def __(mo):
    mo.md(r"""## Predict (What-if-analysis) - Conversion""")
    return


@app.cell
def __(np):
    def predict_conversions_given_clicks(trace, segment_idx, clicks_number_hco):
        """
        Predict the number of conversions given the number of clicks using the posterior samples
        of p_conversion_hco (segment-specific conversion probabilities).
        
        Parameters:
        - trace: The posterior trace from PyMC sampling.
        - segment_idx: The index of the segment for which to predict conversions.
        - clicks_number_hco: The observed number of clicks for which to predict conversions.
        
        Returns:
        - A NumPy array of predicted conversions based on the posterior samples.
        """
        # Extract posterior samples for p_conversion_hco for the given segment
        p_conversion_samples = trace.posterior['p_conversion_segment'].sel(segment=segment_idx).values.flatten()

        # Predict conversions for each sample using a binomial distribution
        predicted_conversions = np.random.binomial(n=clicks_number_hco, p=p_conversion_samples)
        
        return predicted_conversions
    return predict_conversions_given_clicks,


@app.cell
def __(unique_segments):
    unique_segments
    return


@app.cell
def __():

    # clicks_number_hco_2 = 100  # Example number of clicks for which we want to predict conversions
    # predicted_conversions = predict_conversions_given_clicks(trace_hco, unique_segments[12], clicks_number_hco_2)

    # # Output the predicted conversions (posterior distribution)
    # print(f"Predicted conversions (mean): {np.mean(predicted_conversions)}")
    # print(f"Predicted conversions (95% credible interval): {np.percentile(predicted_conversions, [2.5, 97.5])}")
    return


@app.cell
def __(sim_results):
    sim_results
    return


@app.cell
def __(np, predict_conversions_given_clicks, sim_results, trace_hco):
    conversions_mean_list = []
    conversions_lower_list = []
    conversions_upper_list = []
    for r_idx, row_clicks in sim_results.iterrows():
        clicks_r = row_clicks['predicted_clicks_mean']
        segment_r = row_clicks['segment']
        predicted_conversions = predict_conversions_given_clicks(trace_hco, segment_r, clicks_r)
        conversions_mean_list.append(np.mean(predicted_conversions))
        ci_conversion = np.percentile(predicted_conversions, [2.5, 97.5])
        conversions_lower_list.append(ci_conversion[0])
        conversions_upper_list.append(ci_conversion[1])

    sim_results['predicted_conversions_mean'] = conversions_mean_list
    sim_results['predicted_conversions_lower'] = conversions_lower_list
    sim_results['predicted_conversions_upper'] = conversions_upper_list
    return (
        ci_conversion,
        clicks_r,
        conversions_lower_list,
        conversions_mean_list,
        conversions_upper_list,
        predicted_conversions,
        r_idx,
        row_clicks,
        segment_r,
    )


@app.cell
def __(sim_results):
    sim_results
    return


@app.cell
def __(plt, price_mu_hs_test, sim_results, sns):
    plt.figure(figsize=(10, 6))

    # Plot the line for confidence intervals (from lower to upper)
    for row_idx2, row2 in sim_results.iterrows():
        plt.plot([row2['predicted_conversions_lower'], row2['predicted_conversions_upper']], 
                 [row2['segment'], row2['segment']], color='gray')

    # Plot the dot for the predicted mean
    sns.scatterplot(x='predicted_conversions_mean', y='segment', data=sim_results, color='blue', s=100, label='Predicted Conversions Mean')

    # Add labels and title
    plt.xlabel('Predicted Conversions')
    plt.ylabel('Segment')
    plt.title(f'Predicted conversions given budget ({price_mu_hs_test}$)  with Confidence Intervals by Segment')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.show()
    return row2, row_idx2


@app.cell
def __(mo):
    mo.md(
        r"""
        # Conclusion

        * **maximize conversion**: using campaign 936, and targeting the 35 to 39 males will results in the highest expected conversion.
        * **maximize clicks**: Clicks can drive product awareness. Investing in campaign 916, targeting  30 to 34 females would result in the highest clicks. 
        * campaign, segments (age and gender) have a significant impact on CTR, spend (i.e. Cost per click), and conversion rate
        """
    )
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
