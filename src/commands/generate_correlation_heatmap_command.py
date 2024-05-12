from .base_command import BaseCommand
from services import generate_correlation_heatmap


class GenerateCorrelationHeatmapCommand(BaseCommand):

    def add_arguments(self, parser):
        parser.add_argument('--path', type=str, default='./dataset/creditcard_balanced.csv', help='path to save the matrix')
        parser.add_argument('--read_path', type=str, default='./plots/', help='path to dataset')
        parser.add_argument('--lines', type=str, help='number of lines of original data to consider')

    def handle(self, **options):
        path = options['path']
        read_path = options['read_path']
        lines = options['lines']

        args = list(filter(lambda x: x, [path, lines, read_path]))

        generate_correlation_heatmap(*args)





