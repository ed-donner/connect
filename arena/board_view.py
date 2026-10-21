RED = 1
YELLOW = -1
EMPTY = 0

# Self-contained drop trigger. Inlined into an <img onload="..."> attribute, so
# it runs every time a fresh board HTML lands in the DOM — independent of any
# <script> tag (which innerHTML-parsed HTML does not execute) and independent
# of Gradio's js= parameter (whose execution semantics vary across versions).
#
# On first call, it installs window.__c4 with a Web Audio clack synth and a
# Web-Animations-API drop animator. On every call (incl. the first) it kicks
# the current .new-piece element via window.__c4.drop(), which imperatively
# animates it and fires the clack when the animation finishes. Uses only
# single-quoted JS strings so we can safely embed inside a double-quoted
# HTML attribute with no escaping.
_DROP_TRIGGER_JS = (
    "(function(){var w=window;if(!w.__c4){w.__c4={};var ctx=null;"
    "function getCtx(){if(!ctx){try{ctx=new (w.AudioContext||w.webkitAudioContext)();}"
    "catch(e){return null;}}if(ctx.state==='suspended')ctx.resume();return ctx;}"
    "w.addEventListener('click',getCtx);"
    "w.__c4.clack=function(){var c=getCtx();if(!c)return;var t0=c.currentTime;"
    "var nb=c.createBuffer(1,Math.floor(c.sampleRate*0.008),c.sampleRate);"
    "var nd=nb.getChannelData(0);for(var i=0;i<nd.length;i++)nd[i]=Math.random()*2-1;"
    "var ns=c.createBufferSource();ns.buffer=nb;"
    "var nf=c.createBiquadFilter();nf.type='bandpass';nf.frequency.value=2500;nf.Q.value=1.2;"
    "var ng=c.createGain();ng.gain.setValueAtTime(0.6,t0);"
    "ng.gain.exponentialRampToValueAtTime(0.001,t0+0.02);"
    "ns.connect(nf);nf.connect(ng);ng.connect(c.destination);ns.start(t0);"
    "[[1600,0.22],[2200,0.14]].forEach(function(p){"
    "var o=c.createOscillator();var g=c.createGain();"
    "o.type='sine';o.frequency.value=p[0];g.gain.setValueAtTime(p[1],t0);"
    "g.gain.exponentialRampToValueAtTime(0.001,t0+0.16);"
    "o.connect(g);g.connect(c.destination);o.start(t0);o.stop(t0+0.18);});};"
    "w.__c4.drop=function(){var ps=document.querySelectorAll('.new-piece');"
    "if(!ps.length)return;var played=false;ps.forEach(function(p){"
    "var a=p.animate([{transform:'translateY(-300px)'},{transform:'translateY(0)'}],"
    "{duration:500,easing:'cubic-bezier(0.95,0.05,1,0.5)',fill:'forwards'});"
    "a.onfinish=function(){if(!played){played=true;w.__c4.clack();}};});};}"
    "w.__c4.drop();})();"
)


def to_svg(board):
    """
    SVG of the Connect 4 board.

    Pieces have a recessed inner area (concave gradient) and a dark rim so they
    read as real plastic counters. When there's a winner, a glowing yellow line
    passes through the four winning cells.

    A dropped piece is animated + sounds a clack at landing via a small trigger
    mechanism: a 1x1 invisible <img> whose onload attribute carries
    _DROP_TRIGGER_JS (top of this file), so it fires once per render. It uses
    the Web Animations API (reliable) plus a Web Audio clack (no autoplay issues).
    """
    svg = '''
    <div style="display: flex; justify-content: center;">
    <svg width="450" height="420" viewBox="0 0 450 420">
        <defs>
            <radialGradient id="redOuter" cx="0.35" cy="0.28" r="0.9">
                <stop offset="0%" stop-color="#ff8a8a"/>
                <stop offset="55%" stop-color="#d81f1f"/>
                <stop offset="100%" stop-color="#6a0000"/>
            </radialGradient>
            <radialGradient id="redInner" cx="0.72" cy="0.78" r="0.9">
                <stop offset="0%" stop-color="#c82626"/>
                <stop offset="60%" stop-color="#7a0c0c"/>
                <stop offset="100%" stop-color="#2a0000"/>
            </radialGradient>
            <radialGradient id="yellowOuter" cx="0.35" cy="0.28" r="0.9">
                <stop offset="0%" stop-color="#fff4a0"/>
                <stop offset="55%" stop-color="#dcbe1c"/>
                <stop offset="100%" stop-color="#5c4a00"/>
            </radialGradient>
            <radialGradient id="yellowInner" cx="0.72" cy="0.78" r="0.9">
                <stop offset="0%" stop-color="#d8b820"/>
                <stop offset="60%" stop-color="#8a6c08"/>
                <stop offset="100%" stop-color="#2a2000"/>
            </radialGradient>

            <!-- Blue plastic frame: glossy top, darker bottom -->
            <linearGradient id="frameGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="#2a95ea"/>
                <stop offset="12%" stop-color="#0b78d8"/>
                <stop offset="88%" stop-color="#0058b4"/>
                <stop offset="100%" stop-color="#003a84"/>
            </linearGradient>

            <!-- Stand: matches the frame but darker, shaded -->
            <linearGradient id="standGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="#0060bc"/>
                <stop offset="35%" stop-color="#003e86"/>
                <stop offset="100%" stop-color="#00214e"/>
            </linearGradient>

            <mask id="holes">
                <rect x="25" y="25" width="400" height="320" fill="white"/>
                '''
    svg += ''.join(
        f'<circle cx="{x * 50 + 75}" cy="{y * 50 + 60}" r="20" fill="black"/>\n'
        for y in range(6)
        for x in range(7)
    )
    svg += '''
            </mask>
        </defs>

        <!-- Stand with gradient shading -->
        <path d="M0 360 L25 300 H425 L450 360 L425 385 H25 Z" fill="url(#standGrad)"/>
        <!-- Stand top rim highlight (where the two angled faces meet the deck) -->
        <path d="M25 300 H425" stroke="#66bbf0" stroke-opacity="0.55" stroke-width="1"
              fill="none" stroke-linecap="round"/>
        <!-- Stand front-face highlight strip (catches light) -->
        <path d="M28 361 H422" stroke="#2a8ad8" stroke-opacity="0.35" stroke-width="1"
              fill="none" stroke-linecap="round"/>

        <!-- Pieces (shown through the holes) -->
        '''

    for y in range(6):
        for x, cell in enumerate(board.cells[5 - y]):
            if cell == EMPTY:
                continue
            cx = x * 50 + 75
            cy = y * 50 + 60
            is_new = x == board.latest_x and y == (5 - board.latest_y)
            outer = "redOuter" if cell == RED else "yellowOuter"
            inner = "redInner" if cell == RED else "yellowInner"
            rim = "#3a0000" if cell == RED else "#3a2e00"
            recess_shadow = "#2a0000" if cell == RED else "#2a2000"
            recess_light = "#ff9a9a" if cell == RED else "#fff4a0"
            piece_class = "new-piece" if is_new else ""
            # Start off-screen so the piece is never painted at its final
            # position before WAAPI's onload takes over. The animation runs
            # from translateY(-300px) → 0 with fill:'forwards', so after it
            # finishes the group holds at the natural position.
            initial_style = (
                ' style="transform: translateY(-300px);"' if is_new else ""
            )

            svg += f'''
            <g class="{piece_class}"{initial_style}>
                <circle cx="{cx}" cy="{cy}" r="20" fill="url(#{outer})"
                        stroke="{rim}" stroke-width="1.5"/>
                <circle cx="{cx}" cy="{cy}" r="12.5" fill="url(#{inner})"/>
                <path d="M {cx - 12.5} {cy} A 12.5 12.5 0 0 1 {cx + 12.5} {cy}"
                      fill="none" stroke="{recess_shadow}" stroke-width="1.5"
                      opacity="0.85"/>
                <path d="M {cx + 12.5} {cy} A 12.5 12.5 0 0 1 {cx - 12.5} {cy}"
                      fill="none" stroke="{recess_light}" stroke-width="1"
                      opacity="0.45"/>
            </g>
            '''

    svg += '''

        <!-- Blue plastic frame with gradient, holes cut out -->
        <rect x="25" y="25" width="400" height="320" fill="url(#frameGrad)" rx="10" mask="url(#holes)"/>

        <!-- Bevel highlights / shadows on the frame edges (masked so they
             don't paint inside the holes) -->
        <rect x="30" y="27" width="390" height="3" rx="1.5"
              fill="#7ac6ff" opacity="0.55" mask="url(#holes)"/>
        <rect x="30" y="340" width="390" height="3" rx="1.5"
              fill="#001c44" opacity="0.7" mask="url(#holes)"/>
        <rect x="27" y="30" width="3" height="310" rx="1.5"
              fill="#4fa6ea" opacity="0.4" mask="url(#holes)"/>
        <rect x="420" y="30" width="3" height="310" rx="1.5"
              fill="#002a62" opacity="0.55" mask="url(#holes)"/>

        <!-- Hole rims: outer dark ring + inner top-dark / bottom-light arcs
             that read as a recessed socket -->
        '''
    for y in range(6):
        for x in range(7):
            hx = x * 50 + 75
            hy = y * 50 + 60
            svg += (
                f'<circle cx="{hx}" cy="{hy}" r="20" fill="none" '
                f'stroke="#00376e" stroke-width="2"/>\n'
                f'<path d="M {hx - 19} {hy} A 19 19 0 0 1 {hx + 19} {hy}" '
                f'fill="none" stroke="#001530" stroke-width="1.5" opacity="0.6"/>\n'
                f'<path d="M {hx + 19} {hy} A 19 19 0 0 1 {hx - 19} {hy}" '
                f'fill="none" stroke="#8ac8ff" stroke-width="1" opacity="0.25"/>\n'
            )

    # Winning line overlay: layered strokes give a glow without SVG filters
    # (which some Gradio/HTML pipelines sanitize or render inconsistently).
    line_coords = board.winning_line_coords() if board.winner else None
    if line_coords:
        (x1b, y1b), (x2b, y2b) = line_coords
        x1 = x1b * 50 + 75
        y1 = (5 - y1b) * 50 + 60
        x2 = x2b * 50 + 75
        y2 = (5 - y2b) * 50 + 60
        svg += f'''
            <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"
                  stroke="#ffffff" stroke-opacity="0.35" stroke-width="14"
                  stroke-linecap="round"/>
            <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"
                  stroke="#ffff66" stroke-opacity="0.8" stroke-width="8"
                  stroke-linecap="round"/>
            <line class="win-line" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"
                  stroke="#ffffff" stroke-width="4" stroke-linecap="round"/>
        '''

    svg += '</svg>\n'

    # Drop trigger: an invisible 1x1 GIF whose onload fires when the <img> is
    # parsed from innerHTML. The full setup + drop logic is inlined directly
    # in the onload attribute — onload handlers execute reliably across all
    # browsers/frameworks, whereas <script> tags in innerHTML do not.
    if board.latest_x >= 0:
        svg += (
            '<img src="data:image/gif;base64,'
            'R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" '
            'style="display:none" '
            f'onload="{_DROP_TRIGGER_JS}"/>\n'
        )

    svg += '''
</div>
<style>
    .win-line { animation: winPulse 1.2s ease-in-out infinite; }
    @keyframes winPulse {
        0%, 100% { opacity: 0.9; }
        50%      { opacity: 1; stroke-width: 6; }
    }
</style>
'''
    return svg
