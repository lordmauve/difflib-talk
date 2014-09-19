'''
Id:          "$Id: test_diff2html.py,v 1.14 2014/09/12 10:50:28 daniel.pope Exp $"
Copyright:   Copyright (c) 2014 Bank of America Merrill Lynch, All Rights Reserved
Description:
Source: xbiz.rec.prudence.tests.diff2html
'''
import re
from unittest import TestCase, skip
from diff2html import (
    detectChanges, Hunk, HunkParser, HTMLDiffFormatter, FileDiff,
    similar, similarity
)


class TestIterHunks(TestCase):
    def testParseInitPy(self):
        diff = """Index: src/xbiz/rec/prudence/devutils/__init__.py
diff -u src/xbiz/rec/prudence/devutils/__init__.py:1.2 src/xbiz/rec/prudence/devutils/__init__.py:1.3
--- src/xbiz/rec/prudence/devutils/__init__.py:1.2    Fri Feb 21 12:34:31 2014
+++ src/xbiz/rec/prudence/devutils/__init__.py    Mon May 12 06:28:49 2014
@@ -1 +1,5 @@
-
+'''
+Id:          whatever
+Copyright:   Copyright (c) 2014 Bank of America Merrill Lynch, All Rights Reserved
+Category: DevTool
+'''

"""
        p = HunkParser().parse(diff)
        self.assertEqual(len(p), 1)
        self.assertEqual(
            p[0].versions,
            {
                'apath': 'src/xbiz/rec/prudence/devutils/__init__.py',
                'bpath': 'src/xbiz/rec/prudence/devutils/__init__.py',
                'aver': '1.2',
                'bver': '1.3',
            }
        )


class TestSimilarity(TestCase):
    def assertSimilar(self, a, b):
        qs = similarity(a, b)
        print "similarity(%r, %r) = %0.2f" % (a, b, qs)
        self.assertTrue(
            similar(a, b),
            msg='%r is not similar to %r (similarity: %0.2f)' % (a, b, qs)
        )

    def assertDissimilar(self, a, b):
        qs = similarity(a, b)
        print "similarity(%r, %r) = %0.2f" % (a, b, qs)
        self.assertFalse(
            similar(a, b),
            msg='%r is similar to %r (similarity: %0.2f)' % (a, b, qs)
        )

    def test_respaced_commented_string(self):
        self.assertSimilar(
            "    # 'ArcticProductCode',",
            "    'Arctic_Product_Code',"
        )

    def test_changed_number(self):
        self.assertSimilar(
            "    ('607051', 'MLI'),",
            "    ('133920', 'MLI'),"
        )

    def test_different(self):
        """Only has one identifier in common."""
        self.assertDissimilar(
            "    getSourceData(cobDate=None),",
            "    loadRadarTrades(cobDate, 123)"
        )

    def test_different_2(self):
        self.assertDissimilar(
            "        self.assertTablesEqual(",
            "@cpm_settings.environment('unittest')"
        )

    def test_symbols(self):
        self.assertSimilar(')', '])')

    def test_whitespace_change(self):
        self.assertSimilar('foo', '    foo')

#     def test_function_def(self):
#         self.assertDissimilar(
#             '    def panel(self):',
#             '    def cube(self):',
#         )

    def test_long(self):
        """Test addition of a large amount of text."""
        self.assertSimilar(
            "| BookmapGuid | DealId | (int32) Key_Scenario_Tool_Counterparties | src_Coper_Id | tgt_Coper_Id | (int32) Swaps_BU_Area | CPM_Break_Label | (date) COBDate | TaskName |",
            "| BookmapGuid | DealId | (int32) Key_Scenario_Tool_Counterparties | src_Coper_Id | tgt_Coper_Id | (int32) Swaps_BU_Area | CPM_Break_Label | (date) COBDate | TaskName | (int32) IsCFXO | CFXODealId | src_Admin_Id | tgt_Admin_Id | (int32) break_Admin_Id | src_Arctic_Product_Code | tgt_Arctic_Product_Code | (int32) break_Arctic_Product_Code | (int32) break_Coper_Id | src_Master_Id | tgt_Master_Id | (int32) break_Master_Id | src_Netting_Id | tgt_Netting_Id | (int32) break_Netting_Id | src_Risk_Rollup_Coper_Id | tgt_Risk_Rollup_Coper_Id | (int32) break_Risk_Rollup_Coper_Id | src_Risk_Rollup_Legal_Entity | tgt_Risk_Rollup_Legal_Entity | (int32) break_Risk_Rollup_Legal_Entity | (int32) ArcticSysGenIdCount | # pyflakes:ignore tableLiteral"
        )

    def test_commentted_bracket(self):
        """A single-character difference should not be considered significant."""
        self.assertSimilar(
            "            )",
            "#            )",
        )

    def test_commentted_return(self):
        self.assertSimilar(
            "            return",
            "#            return",
        )

    def test_dissimilar_table_manips(self):
        self.assertDissimilar(
            "            tab, rep, qzdb = qz.data.cube.createQztableCube(table, viewSpec=viewSpec, aggmap={})",
            "            return table"
        )

    def test_dissimilar_try(self):
        self.assertDissimilar(
            "        try:",
            "        return listDates(self.Database(), date_format=DEFAULT_DATE_CONVERSION, settings=settings)",
        )

    def test_dissimilar_indented_lines(self):
        self.assertDissimilar(
            "#         resultTabModel = ShowcubeResultTabModel(",
            "        )",
        )

    def test_similar_assignments(self):
        self.assertSimilar(
            "Title='Transaction Reporting Drilldown for FCA-%s' % source.Name(),",
            "Title=source.Title(),"
        )


class DetectChanges(TestCase):
    def assertChanges(self, lines, pattern):
        lines = [(l[0], l[1:]) for l in lines.strip().splitlines()]
        hunk = Hunk(0, 0, lines)
        regex = '^%s$' % pattern.replace('+', r'\+')

        changes = ''.join(detectChanges(hunk))
        assert re.match(regex, changes), \
            "'%s' does not match pattern '%s'" % (changes, pattern)

    def test_merge_hunk(self):
        lines = """
-        return self.table[
-                    (self.table[self.GROUP_BY_COLS[0]]==usiPrefix) &
-                    (self.table[self.GROUP_BY_COLS[1]]==usiValue) &
-                    op(self.table[column], value)
-                ].nRows()
+        preAggTable = self._getPreaggTable(column)
+        res = preAggTable[
+                    (preAggTable[self.GROUP_BY_COLS[0]]==usiPrefix) &
+                    (preAggTable[self.GROUP_BY_COLS[1]]==usiValue) &
+                    op(preAggTable[column], value)
+                ]
+        return sum(res.colToList('count_{0}'.format(self.GROUP_BY_COLS[0])))
"""
        # This is somewhat heuristic, so these patterns are all acceptable
        self.assertChanges(lines, '(+c|-++)ccc(c+|-++)')

    def test_merge_simple(self):
        lines = """
-foo
-bar
-baz
+
+  foo
+
+  bar
+
+  baz
+
"""
        self.assertChanges(lines, '+c+c+c+')

    def test_merge_simple2(self):
        lines = """
-from foo.bar.baz import FooBar
+from foo.bar.baz import BazBonk
"""
        self.assertChanges(lines, 'c')

    def test_merge_whitespace(self):
        lines = '-    \n+'
        self.assertChanges(lines, 'c')

    def test_merge_multiline_whitespace(self):
        lines = '-    \n-    \n+\n+'
        self.assertChanges(lines, 'cc')

    @skip('Not yet implemented')
    def test_merge_scored(self):
        # FIXME: we get a hit on line 1 that means the better match on line 2 isn't seen
        lines = """
- columnNames = table.getSchema().columnNames
- t = table.groupBy(['PartyBookName', 'Deal'], [
+ return aggregate(table, ['PartyBookName', 'Deal'], [
"""
        self.assertChanges(lines, '-c')


class TestHTMLFormat(TestCase):
    def assertHighlighted(self, aplain, bplain, expected, syntax='python'):
        from pygments.lexers import get_lexer_by_name
        from pygments.formatters import HtmlFormatter
        from pygments import highlight

        lexer = get_lexer_by_name(syntax, stripnl=False)
        formatter = HtmlFormatter(nowrap=True)
        ahtml = highlight(aplain, lexer, formatter)
        bhtml = highlight(bplain, lexer, formatter)

        f = HTMLDiffFormatter()
        highlighted = f.highlightChange(
            ahtml, bhtml,
            aplain, bplain
        )

        self.assertEqual(
            highlighted,
            expected
        )

    def testHighlightChanges(self):
        aplain = '''# ui.ComboBox(self.WhereInput, choices=self._WHEREs, editable=True, size=(ui.Size.EXPAND, -2))],'''
        bplain = '''ui.Label("eg Where('Maker') <<startsWith>>'C' or Where('Length') > 15")],'''

        EXPECTED = (
            u'''<span class="c"><span class="delta"># </span>ui.<span class="delta">ComboBox(self.WhereInput, choices=self._WHEREs, editable=True, size=(ui.Size.EXPAND, -2)</span>)],</span>\n''',
            u'''<span class="n">ui</span><span class="o">.</span><span class="n"><span class="delta">Label</span></span><span class="p"><span class="delta">(</span></span><span class="s"><span class="delta">&quot;eg Where(&#39;Maker&#39;) &lt;&lt;startsWith&gt;&gt;&#39;C&#39; or Where(&#39;Length&#39;) &gt; 15&quot;</span></span><span class="p">)],</span>\n'''
        )

        self.assertHighlighted(aplain, bplain, EXPECTED)

    def testHighlightChanges2(self):
        aplain = '''* test `link <http://arm.bankofamerica.com/PreAppSelection.aspx?ReqType=NEW>`_.'''
        bplain = '''* test `link <http://arm.bankofamerica.com/PreAppSelection.aspx?ReqType=NEW>`_.:'''

        EXPECTED = (
            u'<span class="m">*</span> test <span class="s">`link </span><span class="si">&lt;http://arm.bankofamerica.com/PreAppSelection.aspx?ReqType=NEW&gt;</span><span class="s">`_</span>.\n',
            u'<span class="m">*</span> test <span class="s">`link </span><span class="si">&lt;http://arm.bankofamerica.com/PreAppSelection.aspx?ReqType=NEW&gt;</span><span class="s">`_</span>.<span class="delta">:</span>\n'
        )

        self.assertHighlighted(aplain, bplain, EXPECTED, syntax='rst')
