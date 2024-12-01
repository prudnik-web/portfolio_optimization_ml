import plotly.graph_objects as go
import numpy as np
from pandas import DataFrame, concat
from scipy.stats import norm

def plot_prices_graph(df: DataFrame) -> go:

    # Create the figure
    fig = go.Figure()

    # Add the trace with only a line
    fig.add_trace(
        go.Scattergl(
            x=df.index,  # Assuming 'Date' is the index
            y=df['Average Price For All Stocks'],
            mode='lines',  # Line only
            line=dict(color='blue', width=2),  # Line style
            name="Average Stock Price",  # Legend label
            showlegend=True
        )
    )

    # Fit a linear trendline (using numpy.polyfit)
    # Convert the index (dates) to numerical format (e.g., number of days since start)
    x_vals = (df.index - df.index[0]).days
    y_vals = df['Average Price For All Stocks']

    # Use polyfit to get the coefficients for a linear regression
    slope, intercept = np.polyfit(x_vals, y_vals, 1)

    # Calculate the trendline values
    trendline_y = slope * x_vals + intercept

    # Add the trendline trace
    fig.add_trace(
        go.Scattergl(
            x=df.index,
            y=trendline_y,
            mode='lines',  # Trendline as a line
            line=dict(color='red', width=2, dash='dash'),  # Style of the trendline (red and dashed)
            name="Trendline",  # Legend label for trendline
            showlegend=True,
        )
    )

    # Update layout for labels, styling, and height
    fig.update_layout(
        title="Historical Average Stock Prices with Trendline",  # Title for the graph
        xaxis_title="Date",  # Label for X-axis
        yaxis_title="Average Stock Price (USD)",  # Label for Y-axis
        height=700,  # Set the height of the figure to 700px
        xaxis=dict(
            showgrid=True,  # Show grid lines for better readability
            gridcolor='lightgrey',  # Color of grid lines
            tickformat='%Y-%m-%d',  # Format dates on the X-axis
            rangeslider=dict(visible=False),  # Add a range slider for zooming,
            title_font=dict(size=14, family='Arial Black'),

        ),
        yaxis=dict(
            showgrid=True,  # Show grid lines
            gridcolor='lightgrey',  # Color of grid lines
            title_font=dict(size=14, family='Arial Black'),

        ),
        template="plotly_white",  # Clean white background
        legend=dict(
            title="Legend",  # Add title to the legend
            orientation="h",  # Horizontal legend at the top
            x=0.5, y=-0.2, xanchor='center',
            title_font=dict(size=14, family='Arial Black'),

        )
    )

    return fig

def plot_histogram(df: DataFrame) -> go:
    # Assuming 'returns' DataFrame has been calculated with 'Average Return for Stocks'

    # Create the figure for the histogram
    fig = go.Figure()

    # Add the histogram trace
    fig.add_trace(
        go.Histogram(
            x=df['Average Return for Stocks'],  # Data for the histogram
            nbinsx=30,  # Number of bins in the histogram
            name="Stock Returns",  # Legend label for histogram
            opacity=0.75,  # Opacity of the bars
            marker=dict(color='blue')  # Color of the bars
        )
    )

    # Fit a normal distribution to the data for the trendline
    mu, std = norm.fit(df['Average Return for Stocks'])

    # Generate values for the x-axis (range of returns)
    xmin, xmax = min(df['Average Return for Stocks']), max(df['Average Return for Stocks'])
    x_range = np.linspace(xmin, xmax, 100)

    # Compute the corresponding y-values (normal distribution) for the trendline
    y_range = norm.pdf(x_range, mu, std)

    # Add the trendline (normal distribution curve)
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_range * len(df['Average Return for Stocks']) * (xmax - xmin) / 30,  # Scale the normal curve to match histogram's scale
            mode='lines',  # Trendline as a line
            line=dict(color='red', width=2, dash='dash'),  # Style of the trendline (red and dashed)
            name="Trendline (Normal Distribution)",  # Legend label for trendline
            showlegend=True,

        )
    )

    # Update layout for labels, styling, and height
    fig.update_layout(
        title="Stock Returns Distribution with Trendline",  # Title for the graph
        xaxis_title="Average Return for Stocks",  # Label for X-axis
        yaxis_title="Frequency",  # Label for Y-axis
        height=700,  # Set the height of the figure to 700px
        xaxis=dict(
            showgrid=True,  # Show grid lines for better readability
            gridcolor='lightgrey',  # Color of grid lines
            tickformat='.1%',  # Format x-axis as percentage (optional),
            title_font=dict(size=14, family='Arial Black'),

        ),
        yaxis=dict(
            showgrid=True,  # Show grid lines
            gridcolor='lightgrey',  # Color of grid lines,
            title_font=dict(size=14, family='Arial Black'),

        ),
        template="plotly_white",  # Clean white background
        legend=dict(
            title="Legend",  # Add title to the legend
            orientation="h",  # Horizontal legend at the top
            x=0.5, y=-0.2, xanchor='center'
        )
    )

    return fig

def plot_scatter(
        data: dict
        ) -> go:
    
    volatilities_array = data['volatilities_array']
    returns_array = data['returns_array']
    sharpe_array = data['sharpe_array']
    volatility_max_sharpe = data['volatility_max_sharpe']
    returns_max_sharpe = data['returns_max_sharpe']
    num_stocks = data['num_stocks']
    num_simulations = data['num_simulations']
    
    # Create the scatter plot
    fig = go.Figure()

    # Main scatter plot for volatility, returns, and Sharpe ratios
    fig.add_trace(
        go.Scattergl(
            x=volatilities_array, 
            y=returns_array, 
            mode='markers',
            marker=dict(
                size=8,
                color=sharpe_array,  # Color based on Sharpe Ratio
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Sharpe Ratio')
            )
        )
    )

    # Highlight the point with max Sharpe Ratio
    fig.add_trace(
        go.Scattergl(
            x=[volatility_max_sharpe], 
            y=[returns_max_sharpe], 
            mode='markers',
            marker=dict(
                size=10,
                color='red',
                line=dict(width=1, color='black')  # Black edge for contrast
            ), showlegend=False
        )
    )

    # Customize layout
    fig.update_layout(
        title=f'Efficient frontier. {len(num_stocks)} stocks, {num_simulations} observations',
        xaxis=dict(
            title='Volatility',
            title_font=dict(size=14, family='Arial Black'),
            showgrid=True,        # Enable grid lines
            gridcolor='LightGrey',  # Grid line color
            gridwidth=0.5         # Grid line width
        ),
        yaxis=dict(
            title='Return',
            title_font=dict(size=14, family='Arial Black'),
            showgrid=True,
            gridcolor='LightGrey',
            gridwidth=0.5
        ),
        template='plotly_white',
        showlegend=False,
        height=700,
    )

    # Show the plot
    return fig

def pie_chart(
        data: dict,
        threshold: float
        ) -> go:
    
    num_stocks = data['num_stocks']
    all_weights = data['all_weights']

    # Example data
    df = DataFrame({
        'Ticker': num_stocks,
        'Weight': all_weights
    })

    # Separate "small" and "large" slices
    large_slices = df[df['Weight'] >= threshold]
    small_slices = df[df['Weight'] < threshold]

    # Add "Others" row for small slices
    if not small_slices.empty:
        others_weight = small_slices['Weight'].sum()
        others_row = DataFrame({'Ticker': ['Others'], 'Weight': [others_weight]})
        large_slices = concat([large_slices, others_row], ignore_index=True)

    # Create the Pie Chart using go
    fig = go.Figure(data=[go.Pie(
        labels=large_slices['Ticker'],  # Ticker names
        values=large_slices['Weight'],  # Portfolio weights
        hoverinfo='label+percent',  # Display both label and percentage on hover
        textinfo='percent+label',  # Display percentage and label directly on the chart
        pull=[0.05] * len(large_slices),  # Slightly pull out each slice for better readability
    )])

    fig.update_layout(
        title={
            'text': 'Portfolio Allocation',
            'x': 0.3,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        showlegend=True,  # Show the legend
        legend={
            'orientation': 'v',  # Vertical orientation (default)
            'x': 1.1,  # Position legend to the right of the chart
            'y': 1,  # Align legend to the top
            'xanchor': 'left',  # Anchor legend on its left
            'yanchor': 'top',  # Anchor legend to the top
            'title': {'text': 'Tickers'},  # Optional: Title for the legend
            'font': {'size': 10}  # Adjust font size to fit all items
        }
    )


    return fig

def treemap_chart(data: dict, threshold: float) -> go.Figure:
    num_stocks = data['num_stocks']
    all_weights = data['all_weights']

    # Create a DataFrame
    df = DataFrame({
        'Ticker': num_stocks,
        'Weight': all_weights
    })

    # Separate "small" and "large" slices
    large_slices = df[df['Weight'] >= threshold]
    small_slices = df[df['Weight'] < threshold]

    # Add "Others" row for small slices
    if not small_slices.empty:
        others_weight = small_slices['Weight'].sum()
        others_row = DataFrame({'Ticker': ['Others'], 'Weight': [others_weight]})
        large_slices = concat([large_slices, others_row], ignore_index=True)

    # Create the Treemap
    fig = go.Figure(go.Treemap(
        labels=large_slices['Ticker'],      # Stock tickers or "Others"
        parents=[""] * len(large_slices),  # No hierarchical grouping
        values=large_slices['Weight'],     # Portfolio weights
        textinfo="label+percent entry"  # Show label, weight, and percentage
    ))

    # Update layout for better presentation
    fig.update_layout(
        title={
            'text': 'Portfolio Allocation Treemap',
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )

    return fig

def treemap_forecast_chart(df: DataFrame, weights: list, threshold: float) -> go.Figure:
    """
    Process forecasted prices DataFrame and plot a treemap with forecasted returns.
    
    Parameters:
    - df (DataFrame): Forecasted prices DataFrame with stocks as columns.
    - weights (list): Portfolio weights as a list, matching the column order of the DataFrame.
    - threshold (float): Minimum weight to show as an individual slice; others are grouped.
    
    Returns:
    - fig (go.Figure): Treemap with allocation and forecasted returns.
    """
    # Ensure weights length matches the number of columns in df
    if len(weights) != len(df.columns):
        raise ValueError("The length of weights must match the number of stocks in the DataFrame columns.")
    
    # Get the current prices (first row)
    current_prices = df.iloc[0, :]

    # Get the forecasted prices (last row)
    forecasted_prices = df.iloc[-1, :]

    # Calculate forecasted returns
    forecasted_returns = (forecasted_prices - current_prices) / current_prices

    # Create a DataFrame for the treemap
    data = DataFrame({
        'Ticker': df.columns,  # Column names as tickers
        'Weight': weights,     # Portfolio weights from the list
        'Forecasted Return': forecasted_returns.values
    })

    # Separate "small" and "large" slices
    large_slices = data[data['Weight'] >= threshold]
    small_slices = data[data['Weight'] < threshold]

    # Add "Others" row for small slices
    if not small_slices.empty:
        others_weight = small_slices['Weight'].sum()
        others_return = (small_slices['Weight'] * small_slices['Forecasted Return']).sum() / others_weight
        others_row = DataFrame({'Ticker': ['Others'], 'Weight': [others_weight], 'Forecasted Return': [others_return]})
        large_slices = concat([large_slices, others_row], ignore_index=True)

    # Create the Treemap
    fig = go.Figure(go.Treemap(
        labels=large_slices['Ticker'],      # Stock tickers or "Others"
        parents=[""] * len(large_slices),  # No hierarchical grouping
        values=large_slices['Weight'],     # Portfolio weights
        marker=dict(
            colors=large_slices['Forecasted Return'],  # Color by forecasted return
            colorscale='RdYlGn',                      # Red (low) to Green (high)
            colorbar=dict(title="Forecasted Return")  # Add a color bar
        ),
        textinfo="label+percent entry"  # Show label, weight, and percentage
    ))

    # Update layout for better presentation
    fig.update_layout(
        title={
            'text': 'Portfolio Allocation with Forecasted Returns',
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        template="plotly_white"
    )

    return fig
