import datetime
from dash import Dash, dcc, html, dash_table, Input, Output, State, callback
import plotly.express as px
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

if __name__ == '__main__':

    app = Dash(__name__)
    app.layout = html.Div([

        # Upload zone
        dcc.Upload(
            id='upload-image',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            # Allow multiple files to be uploaded
            multiple=True
        ),

        # Centroids
        html.Div(children=[
            html.H5('Centroids',style={"display": "inline"}),
            dcc.RadioItems(['Manual', 'Auto'], 'Auto', inline=True, style={"display": "inline"}),
        ]),

        # K
        html.Div(children=[
            html.H3(' K ', style={'display': 'inline'}),
            dcc.Input(0, style={'display': 'inline'}),
        ], style={'text-align': 'center'}),

        # Image graph
        html.Div(id='output-image-upload'),
    ])


    def parse_contents(contents, filename, date):

        img = Image.open(filename, 'r')
        torch_img = pil_to_tensor(img)
        fig = px.imshow(torch_img.permute(1, 2, 0))

        return html.Div([
            html.H5(filename, style={'text-align': 'center'}),
            dcc.Graph(figure=fig, style={'height': '1000px', 'width': '1000px', 'margin': 'auto'}),
        ])


    @callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),
              State('upload-image', 'last_modified'))
    def update_output(list_of_contents, list_of_names, list_of_dates):
        if list_of_contents is not None:
            children = [
                parse_contents(c, n, d) for c, n, d in
                zip(list_of_contents, list_of_names, list_of_dates)]
            return children

    app.run(debug=True)
