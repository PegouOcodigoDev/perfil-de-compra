import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc


class DashboardApp:
    
    def __init__(self, df_kmeans: pd.DataFrame, df_dbscan: pd.DataFrame, 
                 metrics_kmeans: dict, metrics_dbscan: dict):
        self.df_kmeans = df_kmeans
        self.df_dbscan = df_dbscan
        self.metrics_kmeans = metrics_kmeans
        self.metrics_dbscan = metrics_dbscan
        
        self.app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])
        self._setup_layout()
        self._setup_callbacks()
    
    def _setup_layout(self):
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("📊 Análise de Perfis de Consumidores", 
                           className="text-center mb-2 text-primary"),
                    html.P("Dashboard Interativa - Clusters de Consumidores Semelhantes", 
                          className="text-center text-muted mb-4"),
                ], className="mb-4")
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H4("📈 K-means", className="mb-0 text-center text-white")
                        ], className="bg-primary"),
                        dbc.CardBody([
                            html.H2(f"{self.metrics_kmeans.get('n_clusters', 0)}", 
                                   className="text-center text-primary mb-2"),
                            html.P("Clusters Identificados", className="text-center text-muted mb-3"),
                            html.Hr(),
                            html.P([
                                html.Strong("🎯 Silhouette Score: "), 
                                f"{self.metrics_kmeans.get('silhouette_score', 0):.3f}"
                            ], className="mb-2"),
                            html.P([
                                html.Strong("⚠️ Ruído: "), 
                                f"{self.metrics_kmeans.get('noise_ratio', 0)*100:.1f}%"
                            ], className="mb-2"),
                            html.P([
                                html.Strong("📊 Davies-Bouldin: "), 
                                f"{self.metrics_kmeans.get('davies_bouldin_score', 0):.3f}"
                            ], className="mb-2"),
                            html.P([
                                html.Strong("⚡ Calinski-Harabasz: "), 
                                f"{self.metrics_kmeans.get('calinski_harabasz_score', 0):.1f}"
                            ], className="mb-0")
                        ])
                    ], className="shadow mb-4")
                ], md=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H4("🔷 DBSCAN", className="mb-0 text-center text-white")
                        ], className="bg-success"),
                        dbc.CardBody([
                            html.H2(f"{self.metrics_dbscan.get('n_clusters', 0)}", 
                                   className="text-center text-success mb-2"),
                            html.P("Clusters Identificados", className="text-center text-muted mb-3"),
                            html.Hr(),
                            html.P([
                                html.Strong("🎯 Silhouette Score: "), 
                                f"{self.metrics_dbscan.get('silhouette_score', 0):.3f}"
                            ], className="mb-2"),
                            html.P([
                                html.Strong("⚠️ Ruído: "), 
                                f"{self.metrics_dbscan.get('noise_ratio', 0)*100:.1f}%"
                            ], className="mb-2"),
                            html.P([
                                html.Strong("📊 Davies-Bouldin: "), 
                                f"{self.metrics_dbscan.get('davies_bouldin_score', 0):.3f}"
                            ], className="mb-2"),
                            html.P([
                                html.Strong("⚡ Calinski-Harabasz: "), 
                                f"{self.metrics_dbscan.get('calinski_harabasz_score', 0):.1f}"
                            ], className="mb-0")
                        ])
                    ], className="shadow mb-4")
                ], md=6)
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Label("🔍 Selecione o Método de Clustering:", className="mb-2"),
                            dcc.Dropdown(
                                id='method-selector',
                                options=[
                                    {'label': '📈 K-means - Clustering Particional', 'value': 'kmeans'},
                                    {'label': '🔷 DBSCAN - Clustering por Densidade', 'value': 'dbscan'}
                                ],
                                value='kmeans',
                                clearable=False
                            )
                        ])
                    ], className="shadow mb-4")
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([html.H5("📊 Distribuição dos Clusters", className="mb-0")], 
                                      className="bg-light"),
                        dbc.CardBody([dcc.Graph(id='cluster-distribution')])
                    ], className="shadow mb-4")
                ], md=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([html.H5("💰 Análise de Valor", className="mb-0")], 
                                      className="bg-light"),
                        dbc.CardBody([dcc.Graph(id='value-analysis')])
                    ], className="shadow mb-4")
                ], md=6)
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([html.H5("👥 Perfis de Consumidores por Cluster", className="mb-0 text-white")], 
                                      className="bg-dark"),
                        dbc.CardBody([html.Div(id='cluster-profiles-table')])
                    ], className="shadow mb-4")
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([html.H5("📋 Detalhes dos Consumidores por Cluster", className="mb-0 text-white")], 
                                      className="bg-dark"),
                        dbc.CardBody([html.Div(id='consumer-details')])
                    ], className="shadow mb-4")
                ])
            ])
        ], fluid=True, style={'padding': '20px', 'max-width': '1600px', 'margin': '0 auto'})
    
    def _setup_callbacks(self):
        @self.app.callback(
            [Output('cluster-distribution', 'figure'),
             Output('value-analysis', 'figure'),
             Output('cluster-profiles-table', 'children'),
             Output('consumer-details', 'children')],
            Input('method-selector', 'value')
        )
        def update_dashboard(method):
            df = self.df_kmeans if method == 'kmeans' else self.df_dbscan
            method_name = 'K-means' if method == 'kmeans' else 'DBSCAN'
            
            cluster_counts = df['cluster'].value_counts().sort_index()
            labels = [f'Cluster {c}' for c in cluster_counts.index]
            
            fig_dist = go.Figure(data=[
                go.Bar(x=labels, y=cluster_counts.values,
                      marker=dict(color=px.colors.qualitative.Vivid),
                      text=cluster_counts.values, textposition='outside')
            ])
            fig_dist.update_layout(
                title=f'Distribuição de Tamanho - {method_name}',
                xaxis_title='', yaxis_title='Número de Clientes',
                template='plotly_white', height=400
            )
            
            cluster_stats = df.groupby('cluster').agg({
                'avg_price': 'mean',
                'total_spent': 'mean'
            }).reset_index()
            cluster_stats['label'] = cluster_stats['cluster'].apply(lambda x: f'Cluster {x}')
            
            fig_value = go.Figure()
            fig_value.add_trace(go.Bar(
                name='Preço Médio',
                x=cluster_stats['label'],
                y=cluster_stats['avg_price'],
                marker_color='lightblue'
            ))
            fig_value.add_trace(go.Bar(
                name='Gasto Total Médio',
                x=cluster_stats['label'],
                y=cluster_stats['total_spent'],
                marker_color='darkblue'
            ))
            fig_value.update_layout(
                title=f'Análise de Valor por Cluster - {method_name}',
                barmode='group',
                template='plotly_white',
                height=400
            )
            
            cluster_summary = df.groupby('cluster').agg({
                'user_id': 'count',
                'total_products': 'mean',
                'avg_price': 'mean',
                'avg_discount': 'mean',
                'avg_rating': 'mean',
                'total_spent': 'mean'
            }).reset_index()
            cluster_summary.columns = ['cluster_id', 'tamanho', 'avg_products', 'avg_price', 
                                      'avg_discount', 'avg_rating', 'total_spent']
            cluster_summary['percentage'] = (cluster_summary['tamanho'] / len(df) * 100).round(1)
            cluster_summary = cluster_summary.sort_values('avg_price', ascending=False)
            
            table_rows = [
                html.Tr([
                    html.Th("Cluster", className="bg-primary text-white"),
                    html.Th("Tamanho", className="bg-primary text-white"),
                    html.Th("% Total", className="bg-primary text-white"),
                    html.Th("Produtos/Cliente", className="bg-primary text-white"),
                    html.Th("Preço Médio", className="bg-primary text-white"),
                    html.Th("Desconto Médio", className="bg-primary text-white"),
                    html.Th("Avaliação Média", className="bg-primary text-white")
                ])
            ]
            
            for _, row in cluster_summary.iterrows():
                table_rows.append(html.Tr([
                    html.Td(html.Strong(f"Cluster {int(row['cluster_id'])}")),
                    html.Td(f"{int(row['tamanho']):,}", className="text-center"),
                    html.Td(f"{row['percentage']:.1f}%", className="text-center"),
                    html.Td(f"{row['avg_products']:.1f}", className="text-center"),
                    html.Td(f"₹{row['avg_price']:,.0f}", className="text-center"),
                    html.Td(f"{row['avg_discount']:.1f}%", className="text-center"),
                    html.Td(f"{row['avg_rating']:.2f}", className="text-center")
                ]))
            
            table = dbc.Table([html.Tbody(table_rows)], 
                            bordered=True, hover=True, responsive=True, striped=True)
            
            display_cols = ['user_id', 'total_products', 'avg_price', 'avg_discount', 
                           'avg_rating', 'total_spent']
            if 'favorite_category' in df.columns:
                display_cols.append('favorite_category')
            
            details_sections = []
            for cluster_id in sorted(df['cluster'].unique()):
                cluster_data = df[df['cluster'] == cluster_id].copy()
                
                if 'avg_price' in cluster_data.columns:
                    cluster_data = cluster_data.sort_values('avg_price', ascending=False)
                
                available_cols = [col for col in display_cols if col in cluster_data.columns]
                
                details_sections.append(
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5(f"Cluster {cluster_id} - {len(cluster_data)} consumidores semelhantes", 
                                   className="mb-0 text-white")
                        ], className="bg-info"),
                        dbc.CardBody([
                            dbc.Table.from_dataframe(
                                cluster_data[available_cols].head(10),
                                striped=True,
                                bordered=True,
                                hover=True,
                                responsive=True,
                                className="mt-2"
                            ),
                            html.P(f"Mostrando 10 de {len(cluster_data)} consumidores (ordenados por preço)", 
                                  className="text-muted mt-2")
                        ])
                    ], className="mb-4")
                )
            
            details_content = html.Div(details_sections)
            
            return fig_dist, fig_value, table, details_content
    
    def run(self, debug: bool = True, host: str = '0.0.0.0', port: int = 8050):
        self.app.run(debug=debug, host=host, port=port)

