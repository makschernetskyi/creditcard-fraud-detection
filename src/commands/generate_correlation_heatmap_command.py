from .base_command import BaseCommand
from services import generate_correlation_heatmap


class GenerateCorrelationHeatmapCommand(BaseCommand):

    def add_arguments(self, parser):
        parser.add_argument('--path', type=str, default='./plots/', help='path to save the matrix')
        parser.add_argument('--lines', type=int, help='number of first lines of original data to consider')
        parser.add_argument('--balanced', type=bool, help='boolean value saying if the data sample should be balanced. '
                                                         'mutually exclusive with --lines argument')

    def handle(self, **options):
        kwargs = {
            'path': options['path'],
            'lines': options['lines'],
            'balanced': options['balanced'],
        }

        generate_correlation_heatmap(**kwargs)




