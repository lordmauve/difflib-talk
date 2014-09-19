import re
import logging
import string
from keyword import kwlist
from cgi import escape
from collections import deque, namedtuple, defaultdict, Counter
from difflib import SequenceMatcher
from itertools import izip, islice
from functools import wraps

from pygments import highlight
from pygments.lexers import guess_lexer_for_filename
from pygments.formatters import HtmlFormatter
from pygments.util import ClassNotFound
from pygments.lexers.special import TextLexer
from pygments.lexers.text import YamlLexer



DIFF_CMD_RE = re.compile(r'^diff -u (?P<apath>[^: ]+?)(?::(?P<aver>[\d.]+))? (?P<bpath>[^: ]+?)(?::(?P<bver>\S*))?$')
ALT_DIFF_CMD_RE = re.compile(r'^diff -u -L (?P<path>\S+) -r\s*(?P<ver>[\d.]+)')
DELTA_RE = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@')

# These don't get used in the current implementation; we sync at diff command lines only
# INDEX_RE = re.compile(r'^Index: (\S*)')
# DIFF_HDR = re.compile(r'^(\+\+\+|---) (\S+?)(?::([\d.]+))?')


# Use YAML lexer for proc files
YamlLexer.filenames += ['*.p']

log = logging.getLogger('diff2html')


class Hunk(namedtuple('BaseHunk', 'aoff boff lines')):
    """A section of a diff.

    * ``aoff`` - the line offset of the hunk in file a
    * ``boff`` - the line offset of the hunk in file b
    * ``lines`` - a list of tuples (side, line).
        Each side is a character '+', '-', ' ' etc, the line is the text.

    """
    @property
    def alines(self):
        return [l for side, l in self.lines if side in ' -']

    @property
    def blines(self):
        return [l for side, l in self.lines if side in ' +']


class DiffParseError(Exception):
    """Failed to parse the diff for some reason."""


class HunkParser(object):
    """Implement a finite state machine for parsing a diff."""
    state = 'index'

    def __init__(self):
        self.files = []
        self.file = None
        self.versions = None
        self.lineno = 0

    def parse_diff_command(self, l):
        mo = DIFF_CMD_RE.match(l)
        if mo:
            self.start_file(mo.groupdict())
        else:
            mo = ALT_DIFF_CMD_RE.match(l)
            if mo:
                path = mo.group('path')
                ver = mo.group('ver')
                self.start_file({
                    'apath': path,
                    'bpath': path,
                    'aver': ver,
                    'bver': 'uncommitted',
                })

            else:
                if l.startswith('diff '):
                    raise DiffParseError("Unknown diff command line '%s'" % l)

    def start_file(self, versions):
        """Start reading hunks for a new file described by the versions dict given."""
        self.end_file()
        self.file = quartzName(versions['bpath'])
        self.versions = versions
        self.file_hunks = []
        self.state = self.parse_delta

    def end_file(self):
        if self.file and self.versions:
            if not self.file_hunks:
                raise DiffParseError("%d: Expected diff hunks for file %s" % (self.lineno, self.file))
            self.files.append({
                'file': self.file,
                'versions': self.versions,
                'hunks': self.file_hunks
            })
        self.file = None
        self.versions = None

    def parse_delta(self, l):
        mo = DELTA_RE.match(l)
        if mo:
            aoff, alen, boff, blen = mo.groups()

            alen = int(alen) if alen is not None else 1
            blen = int(blen) if blen is not None else 1

            lines = []
            alines = 0
            blines = 0
            while alines < alen or blines < blen:
                l = self.next_line()
                try:
                    side, l = l[0], l[1:]
                except IndexError:
                    # Possibly we have a diff where trailing whitespace has been stripped
                    side = ' '
                    l = ''

                if side == '\\':
                    # This is some kind of comment line; reviews have been spotted with
                    # comments like '\ No newline at end of file'
                    continue

                if side not in ' -+':
                    raise DiffParseError("%d: Unexpected diff line prefix %r" % (self.lineno, side))

                lines.append((side, l))

                if side in ' -':
                    alines += 1
                if side in ' +':
                    blines += 1

            self.file_hunks.append(Hunk(
                aoff=int(aoff),
                boff=int(boff),
                lines=lines
            ))
        elif l.startswith('diff '):
            self.parse_diff_command(l)

    def next_line(self):
        self.lineno += 1
        return self.lines.popleft()

    def eof(self):
        if self.file:
            if not self.versions:
                raise DiffParseError("%d: Unexpected EOF: expected version string" % self.lineno)
            if not self.file_hunks:
                raise DiffParseError("%d: Unexpected EOF: expected diff hunks" % self.lineno)
        self.end_file()

    def parse(self, diff):
        self.state = self.parse_diff_command
        self.lines = deque(diff.splitlines())

        while self.lines:
            l = self.next_line()
            self.state(l)

        self.eof()
        return [FileDiff(**f) for f in self.files]


class FileNotFound(Exception):
    """Failed to retrieve a named version of a file from CVS."""


def getVCFile(path, rev):
    import subprocess
    log.info("Requesting %s:%s from Mercurial", path, rev)
    args = ['hg', 'cat', '-r', rev, path]
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if err:
        raise FileNotFound(err)
    return out


class HunkMismatch(Exception):
    """Failed to match a hunk."""


def quartzName(filename):
    """Strip the src/ prefix from CVS."""
    return '/' + re.sub(r'^src/+', '', filename.lstrip('/'))


def blank(s):
    return not s or s.isspace()


TOKENISE_RE = re.compile(r'\n|\w+|[^\w\s]')


def tokenise(lines):
    for i, l in enumerate(lines):
        for t in TOKENISE_RE.findall(l):
            if t in ['self', 'cls', '.']:
                continue
            yield t
        yield '\n'


def rfind(seq, elem):
    """Find the index of the last occurrence of elem in seq.

    Raises ValueError if elem is not in seq.

    """
    for i, e in enumerate(reversed(seq)):
        if e == elem:
            return len(seq) - i - 1
    raise ValueError("%s not found" % elem)


def countNewlines(toks):
    try:
        leading = toks.index('\n')
        trailing = len(toks) - rfind(toks, '\n') - 1
    except ValueError:
        leading = 0
        trailing = len(toks)
        newlines = 0
    else:
        newlines = toks.count('\n')

    return leading, newlines, trailing


def iterDiff(a, b, isjunk=None):
    """Diff sequences a and b, yielding tuples (side, items).

    Side is '-' for items only in a, '+' for items only in b,
    and ' ' for items common to both.

    """
    s = SequenceMatcher(a=a, b=b, isjunk=isjunk)
    la = 0
    lb = 0
    for m in s.get_matching_blocks():
        if m.a > la:
            yield '-', a[la:m.a]
        if m.b > lb:
            yield '+', b[lb:m.b]
        la = m.a + m.size
        lb = m.b + m.size
        if m.size:
            yield ' ', a[m.a:la]


def _linesToSides(lines, default):
    for l in lines:
        if not l:
            yield default
        if l in ['-', '+']:
            yield l
        elif l:
            yield 'c'


TOKEN_RE = re.compile('[A-Z]*[a-z0-9]*|[^[a-zA-Z0-9\\s\'",]')
STOPWORDS = set(kwlist) | set(['self', 'cls'])
STOPWORDS_RE = re.compile(r'\b(%s)\b' % '|'.join(STOPWORDS))


def sorensen(a, b):
    """Compute the Sorensen similarity index of two sequences a and b."""
    ca = Counter(a)
    cb = Counter(b)
    denom = sum(ca.itervalues()) + sum(cb.itervalues())

    if not denom:
        return 1.0

    qs = sum((ca & cb).itervalues()) * 2.0 / denom
    return qs


def modified_sorensen(a, b):
    """Compute the a modified Sorensen similarity.

    We use the smaller of the two sets as the denominator, so that proper subsets
    will have high similarity.

    """
    ca = Counter(a)
    cb = Counter(b)
    union = sum((ca | cb).itervalues())
    denom = min(sum(ca.itervalues()), sum(cb.itervalues()), union / 2.0)

    if not denom:
        return 1.0

    qs = sum((ca & cb).itervalues()) * 1.0 / denom
    return qs


def bigrams(line):
    """Get a list of bigram strings in words."""
    #line = STOPWORDS_RE.sub('', line).lower()
    pairs = (a + b for a, b in izip(islice(line, 1, None), line))
    return (p for p in pairs if p != '  ')  # discard consecutive spaces


def trigrams(line):
    """Get a list of trigram strings in words."""
    #line = STOPWORDS_RE.sub('', line).lower()
    line = line.lower()
    triples = izip(
        line,
        islice(line, 1, None),
        islice(line, 2, None)
    )
    return (''.join(t) for t in triples)


def similarity(a, b):
    """Sorensen similarity based on bigram frequencies"""
    # f = lambda s: (t for t in TOKEN_RE.findall(s) if t not in STOPWORDS)
    f = bigrams
    return modified_sorensen(f(a), f(b))


def similar(a, b, threshold=0.65):
    """Return True if lines a and b are similar.

    Currently implemented as a Sorensen similarity index on characters.

    """
    if bool(a.strip()) ^ bool(b.strip()):
        # Blank lines aren't similar to non-blank lines
        return False
    qs = similarity(a, b)
    return qs >= threshold


class LineSideBuffer(list):
    """A list of line sides, indicated by one of '-', '+' and 'c'.

    The add() method allows line sides to be appended, preserving a valid
    order, while ensuring that deletions and insertions are not arbitrarily
    interleaved.

    """

    def reverse_enumerate(self):
        for i in xrange(len(self) - 1, -1, -1):
            yield i, self[i]

    def add(self, c):
        """If we have a -, insert it before all trailing + lines.

        Otherwise, append.

        """
        if c == '-':
            for i, s in self.reverse_enumerate():
                if s != '+':
                    self.insert(i + 1, c)
                    break
            else:
                self.insert(0, c)
                return
        else:
            self.append(c)


def _tokenDiffToSides(diff, alines, blines):
    """Given a diff of a set of lines, identify which are changes.

    """
    # Build a dictionary of which lines match which others
    #
    # For every matching section, record the line in b that
    # contributed a match for the line in a.
    #
    # Also count the total number of lines in each.

    # TODO: currently we only consider the first line in b for
    # each line in a. We could score the match on each line, eg. the total
    # number of chars matched, then search for the best score (greedily)

    match = defaultdict(Counter)  # line no in a -> list of matching line nos in b
    na = nb = 0
    for side, chunk in diff:
        leading, newlines, trailing = countNewlines(chunk)

        if side == '-':
            na += newlines
        elif side == '+':
            nb += newlines
        else:
            chunksize = sum(len(c) for c in chunk)

            if leading:
                match[na][nb] += chunksize
            if newlines:
                for _ in xrange(newlines - 1):
                    na += 1
                    nb += 1
                    match[na][nb] += chunksize
                na += 1
                nb += 1
            if trailing:
                match[na][nb] += chunksize

    out = LineSideBuffer()

#     if any('onTabAdded' in l for l in blines):
#         print "onTabAdded"

    # From our match dictionary, and line counts, reconstruct the
    # list of sides.
    #
    # For each line in a we take the best match in b. If a is ahead,
    # we catch b up by outputting an addition. If there is no match,
    # we output a deletion.
    cb = 0
    for a in xrange(na):
        am = match[a].most_common()

        for m, _ in am:
            if m < cb:
                continue

            # a is ahead, consume lines of b
            while m > cb:
                out.add('+')
                cb += 1

            # If the lines are similar, this is a change
            if similar(alines[a], blines[cb]):
                # changed line
                out.add('c')
                cb += 1
            else:
                # Otherwise, let's call it a deletion
                out.add('-')
            break
        else:
            # If the lines are similar, this is still potentially a match
            # It is just not indicated as such by the match dict
            if cb < nb:
                la, lb = alines[a], blines[cb]
                if not la.strip() and not lb.strip():
                    if not out or out[-1] == 'c':
                        out.add('c')
                        cb += 1
                        continue
                elif similar(la, lb):
                    out.add('c')
                    cb += 1
                    continue

            # No lines in b match this line in a
            out.add('-')

    # Consume the rest of b
    for b in xrange(nb - cb):
        out.add('+')

    return out


def detectChanges(hunk):
    """Detect changed lines in hunk by diffing on tokens.

    """
    out = []
    ac = []
    bc = []

    def merge():
        if not ac:
            out.extend(['+'] * len(bc))
            del bc[:]
            return
        if not bc:
            out.extend(['-'] * len(ac))
            del ac[:]
            return

        atok = list(tokenise(ac))
        btok = list(tokenise(bc))

        diff = iterDiff(atok, btok)
        sides = list(_tokenDiffToSides(diff, ac, bc))

        changed = sides.count('c')
        deleted = sides.count('-')
        added = sides.count('+')
        assert len(ac) == deleted + changed
        assert len(bc) == added + changed

        out.extend(sides)

        del ac[:]
        del bc[:]

    for side, l in hunk.lines:
        if side == ' ':
            merge()
            out.append(side)
        elif side == '-':
            ac.append(l)
        elif side == '+':
            bc.append(l)

    merge()
    return out


def memoize(func):
    """Decorator that caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned, and
    not re-evaluated.
    """
    cache = {}

    @wraps(func)
    def wrapper(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.iteritems()))) if kwargs else (args,)
        try:
            return cache[key]
        except KeyError:
            cache[key] = value = func(*args, **kwargs)
            return value
        except TypeError:
            # uncachable -- for instance, passing a list as an argument.
            # Better to not cache than to blow up entirely.
            return func(*args, **kwargs)

    return wrapper



class FileDiff(object):
    def __init__(self, file, versions, hunks):
        self.file = file
        self.versions = versions
        self.hunks = hunks

    def validateDiff(self, original):
        """Validate that this file diff can cleanly apply to the original lines given."""
        for n, hunk in enumerate(self.hunks):
            start = hunk.aoff - 1
            alines = hunk.alines
            end = start + len(alines)
            original_lines = original[start:end]
            if original_lines != alines:
                raise HunkMismatch('Hunk %d mismatched at offset %d' % (
                    n + 1, hunk.aoff
                ))

    def rediff(self, map_func=string.rstrip):
        """Return a new diff where the lines are compared after mapping.

        The default mapping is string.rstrip, which disregards trailing whitespace.

        """
        a = map(map_func, self.getOriginalLines())
        b = map(map_func, self.getModifiedLines())
        hunks = []
        matcher = SequenceMatcher(a=a, b=b)
        for group in matcher.get_grouped_opcodes(3):
            lines = []
            aoff = group[0][1] + 1
            boff = group[0][3] + 1
            for tag, i1, i2, j1, j2 in group:
                if tag == 'equal':
                    for line in a[i1:i2]:
                        lines.append((' ', line))
                    continue
                if tag == 'replace' or tag == 'delete':
                    for line in a[i1:i2]:
                        lines.append(('-', line))
                if tag == 'replace' or tag == 'insert':
                    for line in b[j1:j2]:
                        lines.append(('+', line))
            hunks.append(Hunk(aoff, boff, lines))

        diff = FileDiff(self.file, self.versions, hunks)
        # Update memoized fields we have already accessed
        mapped_a = '\n'.join(a)
        diff.getOriginal = lambda: mapped_a
        diff.getModifiedLines = lambda: b
        return diff

    @memoize
    def getOriginal(self):
        """Load the original source for the current file."""
        if self.versions['apath'] == '/dev/null':
            ls = ''
        else:
            ls = getVCFile('src' + self.file, self.versions['aver'])
        return ls

    @memoize
    def getOriginalLines(self):
        return self.getOriginal().splitlines()

    @memoize
    def getModifiedLines(self):
        blines = []
        for hunk in self.iterSections():
            if isinstance(hunk, Hunk):
                for side, l in hunk.lines:
                    if side in ' +':
                        blines.append(l)
            else:
                for l in hunk:
                    blines.append(l)
        return blines

    @memoize
    def getModified(self):
        return '\n'.join(self.getModifiedLines())

    def iterSections(self):
        """Iterate over the sections of the file."""
        original = self.getOriginalLines()
        last = 0

        for n, hunk in enumerate(self.hunks):
            start = hunk.aoff - 1

            if start - last:
                yield original[last:start]

            alines = hunk.alines
            end = start + len(alines)
            original_lines = original[start:end]
            if original_lines != alines:
                raise HunkMismatch('%s: Hunk %d mismatched at offset %d\n%r != %r' % (
                    self.file, n + 1, hunk.aoff, alines, original_lines
                ))
            yield hunk
            last = end

        final = original[last:]
        if final:
            yield final


def format_html(filename, source):
    """Format the source text given as HTML with Pygments."""
    try:
        lexer = guess_lexer_for_filename(filename, source, stripnl=False)
    except ClassNotFound:
        lexer = TextLexer(stripnl=False)
    formatter = HtmlFormatter(nowrap=True)
    return highlight(source, lexer, formatter)


class HTMLDiffFormatter(object):
    """Format a diff as HTML."""

    TOK_RE = re.compile('[\s\xa0]{1,4}|\w+|.')

    def getFormatted(self, diff):
        """Format the left and right sides of the text with Pygments."""
        filename = diff.file
        return (
            format_html(filename, diff.getOriginal()),
            format_html(filename, diff.getModified())
        )

    def getFormattedLines(self, diff):
        """Format the left and right sides of the text with Pygments."""
        f = self.getFormatted(diff)
        return tuple(s.splitlines() for s in f)

    def highlightChange(self, a, b, aplain, bplain):
        """Highlight the differences in a changed line.

        a and b are HTML syntax-highlighted lines; aplain and bplain are the
        plain text equivalents.

        This is implemented in three stages:

        1. Diff the plain text.

            The diff is done token by token; this avoids matching, say, bits of
            keywords like 'for' with identifiers like 'form'.

            The output is a stream of plain text tokens and "change markers"
            for each of the a and b sides.

        2. Merge the change markers into the HTML.

            The output of the first phase is used to subdivide the HTML
            and insert the change markers in the right places.

            The output is a stream of HTML tokens and change markers.

        3. Render the result as an HTML string.

            This involves converting the change markers to new HTML tags,
            ensuring that they will be correctly nested in the resulting
            HTML.

        """
        atok = self.TOK_RE.findall(aplain)
        btok = self.TOK_RE.findall(bplain)

        s = SequenceMatcher(a=atok, b=btok, isjunk=blank)

        # Use ints as change markers
        DELTA_ON = 1
        DELTA_OFF = 0

        def change(toks):
            """Add delta tags around toks."""
            return [DELTA_ON] + toks + [DELTA_OFF]

        # Use the differ to insert change markers into the tok stream
        aout = []
        bout = []
        la = 0
        lb = 0
        lena = len(atok)
        lenb = len(btok)
        matches = s.get_matching_blocks()

        # FIXME: Profiling suggests that doing a full diff (with
        # get_matching_blocks) is very slow. We could consider limiting
        # recursion for a faster approximate match.
        # from difflib import Match
        # matches = [s.find_longest_match(0, lena, 0, lenb), Match(len(a), len(b), 0)]
        for m in matches:
            # Disregard small matches in the middle of a line
            if m.a + m.size < lena or m.b + m.size < lenb:
                charsize = sum(len(t) for t in atok[m.a:m.a + m.size])
                if 0 < charsize < 3:
                    continue

            if m.a > la:
                aout.extend(change(atok[la:m.a]))

            if m.b > lb:
                bout.extend(change(btok[lb:m.b]))

            la = m.a + m.size
            lb = m.b + m.size

            if m.size:
                aout.extend(atok[m.a:la])
                bout.extend(btok[m.b:lb])

        def add_tags(intoks, original):
            """Use character counts to merge in the syntax highlighting tags.

            intoks is a list of tokens and change markers from the plain string.
            original is the highlighted string.

            Returns a new string of tokens that may contain HTML tags and
            entities as well as text tokens and change markers.

            """
            out = []
            pos = 0  # Working position in the HTML string
            startpos = 0  # Last position flushed to out

            ol = len(original)

            for tok in intoks:
                try:
                    chars = len(tok)
                except TypeError:
                    # Tok is an integer change marker
                    # The change markers should pass straight through
                    if startpos != pos:
                        out.append(original[startpos:pos])
                        startpos = pos
                    out.append(tok)
                    continue

                # Count out chars text characters from the HTML

                while chars:
                    c = original[pos]
                    if c == '<':
                        for i in xrange(pos + 1, ol):
                            if original[i] == '>':
                                break
                        tagend = i + 1
                        # Emit tags as a single token
                        if startpos != pos:
                            out.append(original[startpos:pos])
                        out.append(original[pos:tagend])
                        pos = startpos = tagend
                    elif c == '&':
                        # Entities in the HTML represent one char of the plain string.
                        while pos < ol:
                            pos += 1
                            if original[pos] == ';':
                                pos += 1
                                break
                        chars -= 1

                    else:
                        # Plain characters in the HTML are 1:1 with the plain string.
                        chars -= 1
                        pos += 1
            if startpos != pos:
                out.append(original[startpos:pos])

            # Output any trailing tags
            if pos < len(original):
                out.append(original[pos:])
            return out

        aout = add_tags(aout, a)
        bout = add_tags(bout, b)

        def flatten(l):
            """Apply the delta tags with correct nesting."""
            delta = 0
            out = []
            buf = []  # buffered deltas

            def pushbuf():
                if buf:
                    out.extend(['<span class="delta">'] + buf + ['</span>'])
                    del buf[:]
            for t in l:
                if t == 0:
                    pushbuf()
                    delta = 0
                    continue
                elif t == 1:
                    delta = 1
                    continue

                if t.startswith('<'):
                    if delta:
                        pushbuf()
                    out.append(t)
                elif delta:
                    buf.append(t)
                else:
                    out.append(t)

            pushbuf()
            return ''.join(out)

        return flatten(aout), flatten(bout)

    def formatNewFile(self, id, diff):
        version = 'New File (%s)' % (
            diff.versions['bver'] or 'uncommitted'
        )
        out = [
            '<table class="diff" id="%s">' % escape(id),
            '<colgroup><col class="num" width="16"><col width="100%"></colgroup>',
            '    <tr><th colspan="2" class="filename">{file}</th></tr>'.format(file=escape(id)),
            '    <tr><th colspan="2">{version}</th></tr>'.format(version=escape(version))
        ]

        formatted = format_html(id, diff.getModified()).splitlines()
        for i, l in enumerate(formatted, start=1):
            out.append(
                '    <tr class="addition"><td class="num">{n}</td><td>{l}</td></tr>'.format(
                    l=l,
                    n=i,
                )
            )
        out.append('</table>')
        return '\n'.join(out)

    def formatDiff(self, diff):
        """Format the given FileDiff as an HTML table."""
        id = quartzName(diff.file)

        if diff.versions['apath'] == '/dev/null':
            return self.formatNewFile(id, diff)

        # Ensure the original text is cached before we do the timed section
        diff.getOriginal()

        alines, blines = self.getFormattedLines(diff)
        aplain = diff.getOriginalLines()
        bplain = diff.getModifiedLines()

        aver = diff.versions['aver'] or 'New file'
        bver = diff.versions['bver'] or 'uncommitted'
        out = [
            '<table class="diff" id="%s">' % escape(id),
            '<colgroup><col class="num" width="16"><col width="50%"><col class="num" width="16"><col width="50%"></colgroup>',
            '    <tr><th colspan="4" class="filename">{file}</th></tr>'.format(file=escape(id)),
            '    <tr><th colspan="2">{aver}</th><th colspan="2">{bver}</th></tr>'.format(aver=aver, bver=bver)
        ]

        def row(a, b, cls=None):
            """Include a row of output.

            a and b are 0-based line numbers, or None to skip that line.

            """
            if a is not None and a < len(alines):
                ahtml = alines[a]
                atext = aplain[a]
                na = a + 1
            else:
                ahtml = na = ''

            if b is not None and b < len(blines):
                bhtml = blines[b]
                btext = bplain[b]
                nb = b + 1
            else:
                bhtml = nb = ''

            if cls == 'change':
                ahtml, bhtml = self.highlightChange(ahtml, bhtml, atext, btext)

            out.append(
                '    <tr{cls}><td class="num">{na}</td><td>{a}</td><td class="num">{nb}</td><td>{b}</td></tr>'.format(
                    a=ahtml,
                    b=bhtml,
                    cls=' class="%s"' % escape(cls) if cls else '',
                    na=na,
                    nb=nb
                )
            )

        a = b = 0

        # Regex and replacement have to be written like this
        # So CVS doesn't replace them.
        #
        # Rewriting your source: worst idea ever.
        #
        cvs_tag_re = re.compile(r'\$' + r'Id.*?\$')
        cvs_tag_repl = '$' + 'Id$'

        def remove_cvs_tag(l):
            return cvs_tag_re.sub(cvs_tag_repl, l)

        def skippable(sides):
            off = 0
            for s in sides:
                if s == 'c':
                    la = remove_cvs_tag(alines[a + off])
                    lb = remove_cvs_tag(blines[b + off])
                    if la != lb:
                        return False
                    off += 1
                elif s == ' ':
                    off += 1
                else:
                    return False
            return True

        state = {
            'collapsing': False
        }

        def start_collapse():
            if not state['collapsing']:
                out.append('<tbody class="common collapsible">')
                state['collapsing'] = True

        def end_collapse():
            if state['collapsing']:
                out.append('</tbody>')
                state['collapsing'] = False

        for hunk in diff.iterSections():
            if isinstance(hunk, Hunk):
                sides = detectChanges(hunk)
                if skippable(sides):
                    start_collapse()
                else:
                    end_collapse()
                for side in sides:
                    if side == ' ':
                        row(a, b, 'common')
                        a += 1
                        b += 1
                    elif side == 'c':
                        row(a, b, 'change')
                        a += 1
                        b += 1
                    elif side == '-':
                        row(a, None, 'deletion')
                        a += 1
                    elif side == '+':
                        row(None, b, 'addition')
                        b += 1
            else:
                collapsible = len(hunk) > 4
                if collapsible:
                    start_collapse()
                for l in hunk:
                    row(a, b)
                    a += 1
                    b += 1

        end_collapse()
        out.append('</table>')
        return '\n'.join(out)


def pre(s):
    return '<pre>%s</pre>' % escape(s)


def preloadAll(fileDiffs, workers=10):
    """Preload all file diffs given in parallel.

    This uses multiprocessing.dummy.Pool (ie. threads) to load the full text
    of files in parallel.

    There would be an advantage to doing some of the formatting in parallel,
    (the first can start formatting before all the others finish loading),
    but this affects the logged timings which are important for now.

    """
    from multiprocessing.dummy import Pool

    # Fix a bug in Python <2.7.4
    import threading
    from weakref import WeakKeyDictionary
    current = threading.current_thread()
    if not hasattr(current, '_children'):
        current._children = WeakKeyDictionary()

    p = Pool(workers)
    return p.map(lambda f: f.getOriginal(), fileDiffs)


def diff2html(diff, ignore_trailing_whitespace=False):
    """Render a review request diff as a syntax-highlighted, side-by-side HTML table."""
    try:
        files = HunkParser().parse(diff)
        preloadAll(files)

        if ignore_trailing_whitespace:
            files = [f.rediff() for f in files]

        formatter = HTMLDiffFormatter()

        diffs = (formatter.formatDiff(f) for f in files)
        return '\n\n'.join(diffs)
    except Exception:
        import traceback
        return pre('Error formatting fancy diff:\n\n' + traceback.format_exc()) + pre(diff)
