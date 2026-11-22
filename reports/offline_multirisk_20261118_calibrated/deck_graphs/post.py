"""Add the charts to the deck with python-pptx (Keynote does not draw pptxgenjs charts; these open in both)."""
import json
import sys
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION, XL_MARKER_STYLE, XL_TICK_LABEL_POSITION, XL_TICK_MARK
from pptx.enum.text import PP_ALIGN
from pptx.enum.dml import MSO_LINE
from pptx.util import Inches, Pt

src, out = sys.argv[1], sys.argv[2]
specs = json.loads((Path(__file__).parent / 'charts.json').read_text())
prs = Presentation(src)
INK, MUTED, GRID = RGBColor.from_string('1B2A34'), RGBColor.from_string('55646D'), RGBColor.from_string('E1E6E9')
FONT = 'Microsoft YaHei'


def style_font(font, size=12, color=MUTED, bold=False):
    font.size, font.name, font.bold = Pt(size), FONT, bold
    font.color.rgb = color


def axis_title(axis, text):
    axis.has_title = True
    tf = axis.axis_title.text_frame
    tf.text = text
    style_font(tf.paragraphs[0].runs[0].font, 12, MUTED)


for sp in specs:
    slide = prs.slides[sp['slide'] - 1]
    cd = CategoryChartData()
    cd.categories = sp['categories']
    for s in sp['series']:
        cd.add_series(s['name'], s['values'])
    kind = {'line': XL_CHART_TYPE.LINE_MARKERS, 'hbar': XL_CHART_TYPE.BAR_CLUSTERED, 'col': XL_CHART_TYPE.COLUMN_CLUSTERED}[sp['kind']]
    gf = slide.shapes.add_chart(kind, Inches(sp['x']), Inches(sp['y']), Inches(sp['w']), Inches(sp['h']), cd)
    gf.name = f"chart-{sp['slide']}-{sp['kind']}"
    ch = gf.chart
    ch.font.name, ch.font.size = FONT, Pt(12)
    ca, va = ch.category_axis, ch.value_axis
    small = sp.get('small', False)
    style_font(ca.tick_labels.font, 9 if small else (13 if sp['kind'] != 'line' else 12), INK if sp['kind'] != 'line' else MUTED)
    style_font(va.tick_labels.font, 9 if small else 12)
    ca.format.line.color.rgb = GRID
    ca.has_major_gridlines = False
    va.minimum_scale = sp.get('min', 0)
    va.maximum_scale = sp['max']
    if sp['kind'] == 'line':
        va.major_unit = sp['unit']
        va.tick_labels.number_format, va.tick_labels.number_format_is_linked = sp['valFmt'], False
        va.has_major_gridlines = True
        va.major_gridlines.format.line.color.rgb = GRID
        va.major_gridlines.format.line.width = Pt(0.75)
        va.format.line.fill.background()
        if sp.get('catTitle'):
            axis_title(ca, sp['catTitle'])
        if sp.get('valTitle'):
            axis_title(va, sp['valTitle'])
        for s, spec in zip(ch.plots[0].series, sp['series']):
            col = RGBColor.from_string(spec['color'])
            s.smooth = False
            if spec.get('dash'):
                s.format.line.color.rgb = col
                s.format.line.width = Pt(1.5)
                s.format.line.dash_style = MSO_LINE.DASH
                s.marker.style = XL_MARKER_STYLE.NONE
                continue
            s.format.line.color.rgb = col
            s.format.line.width = Pt((2.75 if spec.get('bold') else 1.5) if small else (3.5 if spec.get('bold') else 2))
            s.marker.style = XL_MARKER_STYLE.CIRCLE
            s.marker.size = (6 if spec.get('bold') else 4) if small else (9 if spec.get('bold') else 7)
            s.marker.format.fill.solid()
            s.marker.format.fill.fore_color.rgb = col
            s.marker.format.line.color.rgb = col
        ch.has_legend = sp.get('legend', True)
        if ch.has_legend:
            ch.legend.position = XL_LEGEND_POSITION.BOTTOM
            ch.legend.include_in_layout = False
            style_font(ch.legend.font, 12)
    else:
        va.tick_label_position = XL_TICK_LABEL_POSITION.NONE
        va.major_tick_mark = XL_TICK_MARK.NONE
        va.format.line.fill.background()
        ca.major_tick_mark = XL_TICK_MARK.NONE
        va.has_major_gridlines = False
        plot = ch.plots[0]
        plot.gap_width = sp['gap']
        plot.overlap = -10
        plot.has_data_labels = True
        dl = plot.data_labels
        dl.number_format, dl.number_format_is_linked = sp['labelFmt'], False
        dl.position = XL_LABEL_POSITION.OUTSIDE_END
        style_font(dl.font, 10 if small else 12, INK)
        for s, spec in zip(plot.series, sp['series']):
            s.format.fill.solid()
            s.format.fill.fore_color.rgb = RGBColor.from_string(spec['color'])
        if sp.get('pointColors'):
            for i, c in enumerate(sp['pointColors']):
                pt = plot.series[0].points[i]
                pt.format.fill.solid()
                pt.format.fill.fore_color.rgb = RGBColor.from_string(c)
        if sp['kind'] == 'hbar':
            ca.reverse_order = True
        ch.has_legend = False
    if sp.get('title'):
        ch.has_title = True
        tf = ch.chart_title.text_frame
        tf.text = sp['title']
        style_font(tf.paragraphs[0].runs[0].font, 15, INK, True)
    else:
        ch.has_title = False

prs.save(out)
print('charts added:', len(specs), '->', out)
