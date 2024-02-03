import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.metrics import mean_squared_error
import numpy as np
import scipy.stats as stats
import pickle


original_df = pd.read_csv('clean_data.csv')
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
cds=['#9674E4', '#96D7F4']


# Streamlit app
def main():
    st.set_page_config(layout='wide') 
    st.sidebar.title('Diamonds Data Analysis')
    analysis_type = st.sidebar.radio('Choose analysis type:', ['Variable Analysis', 'Regression Model Analysis'])
    
    
    if analysis_type == 'Variable Analysis':
        st.subheader('Dataset Sample')
        head_num = st.radio('Number of displayed rows: ', ['5', '10', '15'], horizontal=True)
        st.write(original_df.head(int(head_num)))
        
        st.sidebar.header('Variable Analysis Options')
        selected_var = st.sidebar.selectbox('Select a variable:', original_df.columns)
        
        if original_df[selected_var].dtype == 'object':
            fig = px.histogram(original_df, x=selected_var, color_discrete_sequence=cds)
            st.plotly_chart(fig)
            
            fig = px.box(original_df, x=selected_var, y='price', color_discrete_sequence=cds)
            st.plotly_chart(fig)
        else:
            fig = px.histogram(original_df, x=selected_var, marginal='rug', hover_data=original_df.columns, color_discrete_sequence=cds)
            st.plotly_chart(fig)
            
            # price vs variable
            fig = px.scatter(original_df, x=selected_var, y='price', trendline='ols', color_discrete_sequence=cds)
            st.plotly_chart(fig)
            
            # plot with kde curve
            # hist_data = [original_df[selected_var]]
            # group_labels = ['distplot'] # name of the dataset
            #
            # fig = ff.create_distplot(hist_data, group_labels, bin_size=.1, curve_type = 'normal')
            # st.plotly_chart(fig)
            
    elif analysis_type == 'Regression Model Analysis':
        st.subheader('Train Dataset Sample')
        st.write(train_df.head())
        
        st.sidebar.header('Regression Model Options')
        model_choice = st.sidebar.radio('Choose a regression model:', ['Forward Selection', 'Backward Elimination'])
        criterion_choice = st.sidebar.selectbox('Select Criterion:', ['AIC', 'BIC', 'adj_rsquare', 'p_value'])
        plot_choice = st.sidebar.radio('Choose a diagnostic plot:', ['Residuals vs Fitted', 'Q-Q Plot', 'Residuals vs Predictors', 'Actual vs Fitted'])
        
        # Load the chosen model
        model_filename = f'models/model_{model_choice.split()[0].lower()}_{criterion_choice.lower()}.pkl'
        model = sm.load(model_filename)
        
        # variables chosen for the regression model
        st.write('Chosen Variables:', model.model.exog_names)
        
        # model performance metrics
        train_preds = model.predict(train_df)
        test_preds = model.predict(test_df)
        train_rmse = np.sqrt(mean_squared_error(train_df['price'], train_preds))
        test_rmse = np.sqrt(mean_squared_error(test_df['price'], test_preds))
        st.write(f'Train Adjusted R²: {model.rsquared_adj:.3f}')
        st.write(f'Train RMSE: {train_rmse:.3f}')
        st.write(f'Test RMSE: {test_rmse:.3f}')
        
        # Regression plots
        residuals = model.resid
        fitted_vals = model.predict()
        if plot_choice == 'Residuals vs Fitted':
            fig = px.scatter(x=fitted_vals, y=residuals, color_discrete_sequence=cds)
            fig.update_layout(title='Residuals vs Fitted', xaxis_title='Fitted values', yaxis_title='Residuals')
            st.plotly_chart(fig)
        elif plot_choice == 'Q-Q Plot':
            qq = stats.probplot(residuals, dist='norm', plot=None)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers', name='Residuals', line=go.scatter.Line(color=cds[0])))
            fig.add_trace(go.Scatter(x=qq[0][0], y=qq[1][1] + qq[1][0]*qq[0][0], mode='lines', name='Fit', line=go.scatter.Line(color=cds[1])))
            fig.update_layout(title='Normal Q-Q', xaxis_title='Theoretical Quantiles', yaxis_title='Standardized Residuals')
            st.plotly_chart(fig)
        elif plot_choice == 'Residuals vs Predictors':
            predictors = model.model.exog_names.copy()
            predictors.remove('Intercept')
            predictor = st.sidebar.selectbox('Select a predictor:', predictors)
            fig = px.scatter(x=train_df[predictor], y=residuals, color_discrete_sequence=cds)
            fig.update_layout(title=f'Residuals vs {predictor}', xaxis_title=predictor, yaxis_title='Residuals')
            st.plotly_chart(fig)
        elif plot_choice == 'Actual vs Fitted':
            # actual vs fitted on either train or test set
            set_choice = st.sidebar.selectbox('Set:', ['Train', 'Test'])
            if set_choice == 'Train':
                fig = px.scatter(x=train_df['price'], y=fitted_vals, color_discrete_sequence=cds)
                fig.add_trace(go.Scatter(x=train_df['price'], y=train_df['price'], mode='lines', name='Identity Line', line=go.scatter.Line(color=cds[1])))
                fig.update_layout(title='Actual vs Fitted', xaxis_title='Actual Price', yaxis_title='Fitted Price')
                st.plotly_chart(fig)
            else:
                fig = px.scatter(x=test_df['price'], y=test_preds, color_discrete_sequence=cds)
                fig.add_trace(go.Scatter(x=test_df['price'], y=test_df['price'], mode='lines', name='Identity Line', line=go.scatter.Line(color=cds[1])))
                fig.update_layout(title='Actual vs Fitted', xaxis_title='Actual Price', yaxis_title='Fitted Price')
                st.plotly_chart(fig)
            
            
if __name__ == '__main__':
    main()
    