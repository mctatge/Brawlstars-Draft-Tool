/**
 * Diagrams for /model. Every figure is a plain inline <svg> — no chart library, no client JS —
 * so they are in the exported HTML, scale with the container, and pick up the page's colour
 * tokens through the `.d` class rules in app/model/model.css (see there for the palette:
 * blue = team A, red = team B, gold = shared weights / meta, green = calibration).
 *
 * Coordinates are hand-laid on a 1120-wide viewBox. The one exception is <SignalShares />,
 * which derives its bar geometry from the weight table so it cannot drift out of sync with
 * backend/bsdraft/engine/scoring.py.
 */

export function StrengthPath() {
  return (
    <svg className="d" viewBox="0 0 1120 486" role="img" aria-label="The strength path: both teams pass through the same embedding table and the same two-layer MLP; only the difference of the two scores survives.">
      <defs>
        <marker id="m1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#333a46"/></marker>
      </defs>
      <line className="spine-l" x1="560" y1="8" x2="560" y2="470"/>

      <text className="hdr a" x="230" y="18" textAnchor="middle">TEAM A · YOUR PICKS</text>
      <text className="hdr b" x="890" y="18" textAnchor="middle">TEAM B · ENEMY PICKS</text>
      <text className="hdr" x="560" y="18" textAnchor="middle">CONTEXT</text>

      <rect className="box-a" x="112" y="32" width="72" height="32"/><text x="148" y="53" textAnchor="middle">a1</text>
      <rect className="box-a" x="194" y="32" width="72" height="32"/><text x="230" y="53" textAnchor="middle">a2</text>
      <rect className="box-a" x="276" y="32" width="72" height="32"/><text x="312" y="53" textAnchor="middle">a3</text>
      <rect className="box-b" x="772" y="32" width="72" height="32"/><text x="808" y="53" textAnchor="middle">b1</text>
      <rect className="box-b" x="854" y="32" width="72" height="32"/><text x="890" y="53" textAnchor="middle">b2</text>
      <rect className="box-b" x="936" y="32" width="72" height="32"/><text x="972" y="53" textAnchor="middle">b3</text>
      <rect className="box" x="482" y="32" width="156" height="32"/><text x="560" y="53" textAnchor="middle">map · mode</text>
      <text className="tiny" x="230" y="80" textAnchor="middle">(N, 3) int64 rows</text>
      <text className="tiny" x="890" y="80" textAnchor="middle">(N, 3) int64 rows</text>

      <path className="wire" d="M148,64 L148,88 L230,88 L230,98" markerEnd="url(#m1)"/>
      <path className="wire" d="M312,64 L312,88 L230,88" />
      <path className="wire" d="M808,64 L808,88 L890,88 L890,98" markerEnd="url(#m1)"/>
      <path className="wire" d="M972,64 L972,88 L890,88" />
      <path className="wire" d="M230,64 L230,88"/><path className="wire" d="M890,64 L890,88"/>

      <rect className="box-k" x="110" y="98" width="240" height="38"/><text x="230" y="122" textAnchor="middle">brawler.weight -&gt; (N,3,32)</text>
      <rect className="box-k" x="770" y="98" width="240" height="38"/><text x="890" y="122" textAnchor="middle">brawler.weight -&gt; (N,3,32)</text>
      <line className="tie" x1="350" y1="117" x2="770" y2="117"/>
      <rect className="chip" x="486" y="107" width="148" height="20"/>
      <text className="tiny gold-t" x="560" y="121" textAnchor="middle">ONE SHARED 108 × 32 TABLE</text>

      <path className="wire" d="M230,136 L230,160" markerEnd="url(#m1)"/>
      <path className="wire" d="M890,136 L890,160" markerEnd="url(#m1)"/>

      <rect className="box-a" x="126" y="160" width="208" height="34"/><text x="230" y="182" textAnchor="middle">mean over slots (N,32)</text>
      <rect className="box-b" x="786" y="160" width="208" height="34"/><text x="890" y="182" textAnchor="middle">mean over slots (N,32)</text>
      <rect className="box-k" x="452" y="160" width="216" height="34"/><text x="560" y="182" textAnchor="middle">ctx (N,24)</text>
      <text className="tiny" x="560" y="152" textAnchor="middle">map_emb 114×16 · mode_emb 7×8</text>

      <path className="wire" d="M230,194 L230,218" markerEnd="url(#m1)"/>
      <path className="wire" d="M890,194 L890,218" markerEnd="url(#m1)"/>
      <path className="wire" d="M470,194 L470,206 L290,206 L290,218" markerEnd="url(#m1)"/>
      <path className="wire" d="M650,194 L650,206 L830,206 L830,218" markerEnd="url(#m1)"/>

      <rect className="box-a" x="140" y="218" width="180" height="34"/><text x="230" y="240" textAnchor="middle">concat ctx (N,56)</text>
      <rect className="box-b" x="800" y="218" width="180" height="34"/><text x="890" y="240" textAnchor="middle">concat ctx (N,56)</text>

      <path className="wire" d="M230,252 L230,276" markerEnd="url(#m1)"/>
      <path className="wire" d="M890,252 L890,276" markerEnd="url(#m1)"/>

      <rect className="box-k" x="100" y="276" width="260" height="54"/>
      <text x="230" y="297" textAnchor="middle">Linear 56 -&gt; 64 · ReLU</text>
      <text x="230" y="317" textAnchor="middle">Linear 64 -&gt; 1</text>
      <rect className="box-k" x="760" y="276" width="260" height="54"/>
      <text x="890" y="297" textAnchor="middle">Linear 56 -&gt; 64 · ReLU</text>
      <text x="890" y="317" textAnchor="middle">Linear 64 -&gt; 1</text>
      <line className="tie" x1="360" y1="303" x2="760" y2="303"/>
      <rect className="chip" x="470" y="293" width="180" height="20"/>
      <text className="tiny gold-t" x="560" y="307" textAnchor="middle">SAME 3,713 WEIGHTS</text>

      <path className="wire" d="M230,330 L230,354" markerEnd="url(#m1)"/>
      <path className="wire" d="M890,330 L890,354" markerEnd="url(#m1)"/>

      <rect className="box-a" x="170" y="354" width="120" height="32"/><text className="big a" x="230" y="376" textAnchor="middle">S(A, c)</text>
      <rect className="box-b" x="830" y="354" width="120" height="32"/><text className="big b" x="890" y="376" textAnchor="middle">S(B, c)</text>

      <path className="wire" d="M230,386 L230,404 L490,404 L490,418" markerEnd="url(#m1)"/>
      <path className="wire" d="M890,386 L890,404 L630,404 L630,418" markerEnd="url(#m1)"/>
      <rect className="box" x="430" y="418" width="260" height="38"/>
      <text className="big" x="560" y="443" textAnchor="middle">S(A, c) − S(B, c)</text>
    </svg>
  );
}

export function CounterPath() {
  return (
    <svg className="d" viewBox="0 0 1120 384" role="img" aria-label="The counter path: each team's attacker vectors are dotted against the other team's defender vectors, and the two crossed products are subtracted.">
      <defs>
        <marker id="m2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#333a46"/></marker>
        <marker id="m2a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#3b82f6"/></marker>
        <marker id="m2b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#ff3b30"/></marker>
      </defs>
      <line className="spine-l" x1="560" y1="8" x2="560" y2="370"/>

      <text className="hdr a" x="230" y="18" textAnchor="middle">TEAM A</text>
      <text className="hdr b" x="890" y="18" textAnchor="middle">TEAM B</text>

      <rect className="box-a" x="112" y="32" width="72" height="32"/><text x="148" y="53" textAnchor="middle">a1</text>
      <rect className="box-a" x="194" y="32" width="72" height="32"/><text x="230" y="53" textAnchor="middle">a2</text>
      <rect className="box-a" x="276" y="32" width="72" height="32"/><text x="312" y="53" textAnchor="middle">a3</text>
      <rect className="box-b" x="772" y="32" width="72" height="32"/><text x="808" y="53" textAnchor="middle">b1</text>
      <rect className="box-b" x="854" y="32" width="72" height="32"/><text x="890" y="53" textAnchor="middle">b2</text>
      <rect className="box-b" x="936" y="32" width="72" height="32"/><text x="972" y="53" textAnchor="middle">b3</text>

      <path className="wire" d="M230,64 L230,78 L163,78 L163,96" markerEnd="url(#m2)"/>
      <path className="wire" d="M230,78 L297,78 L297,96" markerEnd="url(#m2)"/>
      <path className="wire" d="M890,64 L890,78 L823,78 L823,96" markerEnd="url(#m2)"/>
      <path className="wire" d="M890,78 L957,78 L957,96" markerEnd="url(#m2)"/>

      <rect className="box-a" x="104" y="96" width="118" height="38"/><text className="big a" x="163" y="121" textAnchor="middle">PA</text>
      <rect className="box-a" x="238" y="96" width="118" height="38"/><text className="big a" x="297" y="121" textAnchor="middle">QA</text>
      <rect className="box-b" x="764" y="96" width="118" height="38"/><text className="big b" x="823" y="121" textAnchor="middle">PB</text>
      <rect className="box-b" x="898" y="96" width="118" height="38"/><text className="big b" x="957" y="121" textAnchor="middle">QB</text>
      <text className="tiny" x="163" y="88" textAnchor="middle">counter_p · SUM</text>
      <text className="tiny" x="297" y="88" textAnchor="middle">counter_q · SUM</text>
      <text className="tiny" x="823" y="88" textAnchor="middle">counter_p · SUM</text>
      <text className="tiny" x="957" y="88" textAnchor="middle">counter_q · SUM</text>

      <path className="wire wire-a" d="M297,134 L297,170 L630,170 L630,250" markerEnd="url(#m2a)"/>
      <path className="wire wire-b" d="M957,134 L957,192 L490,192 L490,250" markerEnd="url(#m2b)"/>
      <path className="wire wire-a" d="M163,134 L163,214 L430,214 L430,250" markerEnd="url(#m2a)"/>
      <path className="wire wire-b" d="M823,134 L823,228 L690,228 L690,250" markerEnd="url(#m2b)"/>

      <rect className="chip" x="380" y="161" width="150" height="18"/><text className="tiny a-t" x="455" y="174" textAnchor="middle">OUR DEFENCE (16)</text>
      <rect className="chip" x="640" y="183" width="160" height="18"/><text className="tiny b-t" x="720" y="196" textAnchor="middle">THEIR DEFENCE (16)</text>
      <rect className="chip" x="228" y="205" width="150" height="18"/><text className="tiny a-t" x="303" y="218" textAnchor="middle">OUR ATTACK (16)</text>
      <rect className="chip" x="700" y="219" width="156" height="18"/><text className="tiny b-t" x="778" y="232" textAnchor="middle">THEIR ATTACK (16)</text>

      <rect className="box" x="378" y="250" width="164" height="38"/><text className="big" x="460" y="275" textAnchor="middle"><tspan className="a">PA</tspan> · <tspan className="b">QB</tspan></text>
      <rect className="box" x="578" y="250" width="164" height="38"/><text className="big" x="660" y="275" textAnchor="middle"><tspan className="b">PB</tspan> · <tspan className="a">QA</tspan></text>

      <path className="wire" d="M460,288 L460,300 L500,300 L500,314" markerEnd="url(#m2)"/>
      <path className="wire" d="M660,288 L660,300 L620,300 L620,314" markerEnd="url(#m2)"/>
      <rect className="box" x="430" y="314" width="260" height="38"/>
      <text className="big" x="560" y="339" textAnchor="middle">PA·QB − PB·QA</text>
    </svg>
  );
}

export function LogitJoin() {
  return (
    <svg className="d" viewBox="0 0 1120 196" role="img" aria-label="The two halves are added into a single logit, which a sigmoid turns into the probability that team A wins.">
      <defs><marker id="m3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#333a46"/></marker></defs>
      <rect className="box" x="180" y="16" width="280" height="38"/><text className="big" x="320" y="41" textAnchor="middle">S(A,c) − S(B,c)</text>
      <rect className="box" x="660" y="16" width="280" height="38"/><text className="big" x="800" y="41" textAnchor="middle">PA·QB − PB·QA</text>
      <text className="tiny" x="320" y="68" textAnchor="middle">HOW GOOD, IN THIS CONTEXT</text>
      <text className="tiny" x="800" y="68" textAnchor="middle">WHO BEATS WHOM</text>
      <path className="wire" d="M320,54 L320,90 L542,90" markerEnd="url(#m3)"/>
      <path className="wire" d="M800,54 L800,90 L578,90" markerEnd="url(#m3)"/>
      <circle className="node" cx="560" cy="90" r="17"/><text className="big" x="560" y="96" textAnchor="middle">+</text>
      <path className="wire" d="M560,107 L560,120 L395,120 L395,134" markerEnd="url(#m3)"/>
      <rect className="box" x="330" y="134" width="130" height="38"/><text className="big" x="395" y="159" textAnchor="middle">ℓ</text>
      <path className="wire" d="M460,153 L496,153" markerEnd="url(#m3)"/>
      <rect className="box-k" x="500" y="134" width="70" height="38"/><text className="big" x="535" y="159" textAnchor="middle">σ</text>
      <path className="wire" d="M570,153 L606,153" markerEnd="url(#m3)"/>
      <rect className="box" x="610" y="134" width="200" height="38"/><text className="big" x="710" y="159" textAnchor="middle">P(A beats B)</text>
    </svg>
  );
}

export function ExtraRows() {
  return (
    <svg className="d" viewBox="0 0 1120 384" role="img" aria-label="Three embedding tables, each with one extra row: the brawler table's extra row is a trained mask, while the map and mode tables reserve row zero for an unknown bucket that is never trained and no longer reachable.">
      <text className="hdr" x="220" y="20" textAnchor="middle">BRAWLER.WEIGHT</text>
      <text className="hdr" x="560" y="20" textAnchor="middle">MAP_EMB.WEIGHT</text>
      <text className="hdr" x="900" y="20" textAnchor="middle">MODE_EMB.WEIGHT</text>
      <text className="tiny" x="220" y="38" textAnchor="middle">108 × 32</text>
      <text className="tiny" x="560" y="38" textAnchor="middle">114 × 16</text>
      <text className="tiny" x="900" y="38" textAnchor="middle">7 × 8</text>

      <rect className="mtx" x="120" y="48" width="200" height="190"/>
      <rect className="mtx" x="460" y="48" width="200" height="190"/>
      <rect className="mtx" x="800" y="48" width="200" height="190"/>
      <g className="mrow">
        <line x1="120" y1="70" x2="320" y2="70"/><line x1="120" y1="92" x2="320" y2="92"/><line x1="120" y1="114" x2="320" y2="114"/><line x1="120" y1="136" x2="320" y2="136"/><line x1="120" y1="158" x2="320" y2="158"/><line x1="120" y1="180" x2="320" y2="180"/><line x1="120" y1="202" x2="320" y2="202"/>
        <line x1="460" y1="70" x2="660" y2="70"/><line x1="460" y1="92" x2="660" y2="92"/><line x1="460" y1="114" x2="660" y2="114"/><line x1="460" y1="136" x2="660" y2="136"/><line x1="460" y1="158" x2="660" y2="158"/><line x1="460" y1="180" x2="660" y2="180"/><line x1="460" y1="202" x2="660" y2="202"/>
        <line x1="800" y1="70" x2="1000" y2="70"/><line x1="800" y1="92" x2="1000" y2="92"/><line x1="800" y1="114" x2="1000" y2="114"/><line x1="800" y1="136" x2="1000" y2="136"/><line x1="800" y1="158" x2="1000" y2="158"/><line x1="800" y1="180" x2="1000" y2="180"/><line x1="800" y1="202" x2="1000" y2="202"/>
      </g>

      <rect className="hl-a" x="120" y="212" width="200" height="26"/>
      <text className="rowlab a" x="220" y="230" textAnchor="middle">row 107 · MASK</text>
      <rect className="hl-g" x="460" y="48" width="200" height="26"/>
      <text className="rowlab gold-t" x="560" y="66" textAnchor="middle">row 0 · UNKNOWN</text>
      <rect className="hl-g" x="800" y="48" width="200" height="26"/>
      <text className="rowlab gold-t" x="900" y="66" textAnchor="middle">row 0 · UNKNOWN</text>

      <text className="cap" x="120" y="266">107 real brawlers, densely indexed.</text>
      <text className="cap" x="120" y="284">The extra row is <tspan className="a">trained</tspan>: masked into 30% of</text>
      <text className="cap" x="120" y="302">rows, then used to pad short teams.</text>
      <text className="cap" x="460" y="266">113 real ranked maps at rows 1–113.</text>
      <text className="cap" x="460" y="284">Row 0 is <tspan className="gold-t">never trained</tspan> — non-ranked</text>
      <text className="cap" x="460" y="302">rows are dropped before training.</text>
      <text className="cap" x="800" y="266">6 ranked modes at rows 1–6.</text>
      <text className="cap" x="800" y="284">Row 0 is <tspan className="gold-t">never trained</tspan>, and now</text>
      <text className="cap" x="800" y="302">unreachable: pinned vocab routes</text>
      <text className="cap" x="800" y="320">unknowns elsewhere.</text>

      <rect className="strip" x="120" y="336" width="880" height="40"/>
      <text className="cap" x="140" y="353">AND ONE MORE, ADDED AT LOAD TIME: serve.py appends a column-mean "average" row to every table, and clamps</text>
      <text className="cap" x="140" y="369">any unrecognised id onto it. In memory the tables are 109, 115 and 8 rows — a brawler released after training scores as average, not as Shelly.</text>
    </svg>
  );
}

export function DecayClocks() {
  return (
    <svg className="d" viewBox="0 0 1120 300" role="img" aria-label="Two exponential decay curves: the model's 30-day half-life and the empirical statistics' 21-day half-life.">
      <defs><marker id="m5" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#333a46"/></marker></defs>
      <line className="ax" x1="100" y1="248" x2="1060" y2="248"/>
      <line className="ax" x1="100" y1="40" x2="100" y2="248"/>
      <line className="grid" x1="100" y1="148" x2="1060" y2="148"/>
      <line className="grid" x1="100" y1="48" x2="1060" y2="48"/>
      <text className="tiny" x="92" y="52" textAnchor="end">1.0</text>
      <text className="tiny" x="92" y="152" textAnchor="end">0.5</text>
      <text className="tiny" x="92" y="252" textAnchor="end">0.0</text>
      <text className="tiny" x="100" y="272" textAnchor="middle">today</text>
      <text className="tiny" x="340" y="272" textAnchor="middle">30d</text>
      <text className="tiny" x="580" y="272" textAnchor="middle">60d</text>
      <text className="tiny" x="820" y="272" textAnchor="middle">90d</text>
      <text className="tiny" x="1060" y="272" textAnchor="middle">120d</text>
      <text className="lbl" x="580" y="294" textAnchor="middle">age of the match</text>

      <polyline className="curve-a" points="100.0,48.0 104.0,50.3 108.0,52.6 112.0,54.8 116.0,57.0 120.0,59.2 124.0,61.4 128.0,63.5 132.0,65.7 136.0,67.7 140.0,69.8 144.0,71.9 148.0,73.9 152.0,75.9 156.0,77.9 160.0,79.8 164.0,81.8 168.0,83.7 172.0,85.5 176.0,87.4 180.0,89.3 184.0,91.1 188.0,92.9 192.0,94.7 196.0,96.4 200.0,98.2 204.0,99.9 208.0,101.6 212.0,103.3 216.0,104.9 220.0,106.6 224.0,108.2 228.0,109.8 232.0,111.4 236.0,113.0 240.0,114.5 244.0,116.0 248.0,117.6 252.0,119.1 256.0,120.5 260.0,122.0 264.0,123.5 268.0,124.9 272.0,126.3 276.0,127.7 280.0,129.1 284.0,130.4 288.0,131.8 292.0,133.1 296.0,134.4 300.0,135.8 304.0,137.0 308.0,138.3 312.0,139.6 316.0,140.8 320.0,142.1 324.0,143.3 328.0,144.5 332.0,145.7 336.0,146.8 340.0,148.0 344.0,149.1 348.0,150.3 352.0,151.4 356.0,152.5 360.0,153.6 364.0,154.7 368.0,155.8 372.0,156.8 376.0,157.9 380.0,158.9 384.0,159.9 388.0,160.9 392.0,161.9 396.0,162.9 400.0,163.9 404.0,164.9 408.0,165.8 412.0,166.8 416.0,167.7 420.0,168.6 424.0,169.5 428.0,170.4 432.0,171.3 436.0,172.2 440.0,173.1 444.0,173.9 448.0,174.8 452.0,175.6 456.0,176.5 460.0,177.3 464.0,178.1 468.0,178.9 472.0,179.7 476.0,180.5 480.0,181.3 484.0,182.0 488.0,182.8 492.0,183.5 496.0,184.3 500.0,185.0 504.0,185.7 508.0,186.4 512.0,187.1 516.0,187.8 520.0,188.5 524.0,189.2 528.0,189.9 532.0,190.6 536.0,191.2 540.0,191.9 544.0,192.5 548.0,193.2 552.0,193.8 556.0,194.4 560.0,195.0 564.0,195.6 568.0,196.2 572.0,196.8 576.0,197.4 580.0,198.0 584.0,198.6 588.0,199.1 592.0,199.7 596.0,200.3 600.0,200.8 604.0,201.3 608.0,201.9 612.0,202.4 616.0,202.9 620.0,203.5 624.0,204.0 628.0,204.5 632.0,205.0 636.0,205.5 640.0,206.0 644.0,206.4 648.0,206.9 652.0,207.4 656.0,207.9 660.0,208.3 664.0,208.8 668.0,209.2 672.0,209.7 676.0,210.1 680.0,210.5 684.0,211.0 688.0,211.4 692.0,211.8 696.0,212.2 700.0,212.6 704.0,213.1 708.0,213.5 712.0,213.8 716.0,214.2 720.0,214.6 724.0,215.0 728.0,215.4 732.0,215.8 736.0,216.1 740.0,216.5 744.0,216.9 748.0,217.2 752.0,217.6 756.0,217.9 760.0,218.3 764.0,218.6 768.0,218.9 772.0,219.3 776.0,219.6 780.0,219.9 784.0,220.3 788.0,220.6 792.0,220.9 796.0,221.2 800.0,221.5 804.0,221.8 808.0,222.1 812.0,222.4 816.0,222.7 820.0,223.0 824.0,223.3 828.0,223.6 832.0,223.9 836.0,224.1 840.0,224.4 844.0,224.7 848.0,224.9 852.0,225.2 856.0,225.5 860.0,225.7 864.0,226.0 868.0,226.2 872.0,226.5 876.0,226.7 880.0,227.0 884.0,227.2 888.0,227.5 892.0,227.7 896.0,227.9 900.0,228.2 904.0,228.4 908.0,228.6 912.0,228.8 916.0,229.1 920.0,229.3 924.0,229.5 928.0,229.7 932.0,229.9 936.0,230.1 940.0,230.3 944.0,230.5 948.0,230.7 952.0,230.9 956.0,231.1 960.0,231.3 964.0,231.5 968.0,231.7 972.0,231.9 976.0,232.1 980.0,232.3 984.0,232.4 988.0,232.6 992.0,232.8 996.0,233.0 1000.0,233.1 1004.0,233.3 1008.0,233.5 1012.0,233.6 1016.0,233.8 1020.0,234.0 1024.0,234.1 1028.0,234.3 1032.0,234.4 1036.0,234.6 1040.0,234.8 1044.0,234.9 1048.0,235.1 1052.0,235.2 1056.0,235.4 1060.0,235.5"/>
      <polyline className="curve-g" points="100.0,48.0 104.0,51.3 108.0,54.5 112.0,57.7 116.0,60.8 120.0,63.8 124.0,66.9 128.0,69.8 132.0,72.7 136.0,75.6 140.0,78.4 144.0,81.2 148.0,83.9 152.0,86.6 156.0,89.3 160.0,91.9 164.0,94.4 168.0,96.9 172.0,99.4 176.0,101.8 180.0,104.2 184.0,106.6 188.0,108.9 192.0,111.2 196.0,113.4 200.0,115.6 204.0,117.8 208.0,119.9 212.0,122.0 216.0,124.1 220.0,126.1 224.0,128.1 228.0,130.1 232.0,132.0 236.0,133.9 240.0,135.8 244.0,137.6 248.0,139.4 252.0,141.2 256.0,142.9 260.0,144.6 264.0,146.3 268.0,148.0 272.0,149.6 276.0,151.2 280.0,152.8 284.0,154.4 288.0,155.9 292.0,157.4 296.0,158.9 300.0,160.4 304.0,161.8 308.0,163.2 312.0,164.6 316.0,166.0 320.0,167.3 324.0,168.6 328.0,169.9 332.0,171.2 336.0,172.5 340.0,173.7 344.0,174.9 348.0,176.1 352.0,177.3 356.0,178.4 360.0,179.6 364.0,180.7 368.0,181.8 372.0,182.9 376.0,184.0 380.0,185.0 384.0,186.0 388.0,187.0 392.0,188.0 396.0,189.0 400.0,190.0 404.0,190.9 408.0,191.9 412.0,192.8 416.0,193.7 420.0,194.6 424.0,195.5 428.0,196.3 432.0,197.2 436.0,198.0 440.0,198.8 444.0,199.6 448.0,200.4 452.0,201.2 456.0,202.0 460.0,202.7 464.0,203.5 468.0,204.2 472.0,204.9 476.0,205.6 480.0,206.3 484.0,207.0 488.0,207.7 492.0,208.3 496.0,209.0 500.0,209.6 504.0,210.2 508.0,210.9 512.0,211.5 516.0,212.1 520.0,212.6 524.0,213.2 528.0,213.8 532.0,214.4 536.0,214.9 540.0,215.4 544.0,216.0 548.0,216.5 552.0,217.0 556.0,217.5 560.0,218.0 564.0,218.5 568.0,219.0 572.0,219.5 576.0,219.9 580.0,220.4 584.0,220.8 588.0,221.3 592.0,221.7 596.0,222.2 600.0,222.6 604.0,223.0 608.0,223.4 612.0,223.8 616.0,224.2 620.0,224.6 624.0,225.0 628.0,225.4 632.0,225.7 636.0,226.1 640.0,226.5 644.0,226.8 648.0,227.2 652.0,227.5 656.0,227.8 660.0,228.2 664.0,228.5 668.0,228.8 672.0,229.1 676.0,229.4 680.0,229.7 684.0,230.0 688.0,230.3 692.0,230.6 696.0,230.9 700.0,231.2 704.0,231.5 708.0,231.7 712.0,232.0 716.0,232.3 720.0,232.5 724.0,232.8 728.0,233.0 732.0,233.3 736.0,233.5 740.0,233.7 744.0,234.0 748.0,234.2 752.0,234.4 756.0,234.6 760.0,234.9 764.0,235.1 768.0,235.3 772.0,235.5 776.0,235.7 780.0,235.9 784.0,236.1 788.0,236.3 792.0,236.5 796.0,236.7 800.0,236.9 804.0,237.0 808.0,237.2 812.0,237.4 816.0,237.6 820.0,237.7 824.0,237.9 828.0,238.1 832.0,238.2 836.0,238.4 840.0,238.6 844.0,238.7 848.0,238.9 852.0,239.0 856.0,239.2 860.0,239.3 864.0,239.4 868.0,239.6 872.0,239.7 876.0,239.9 880.0,240.0 884.0,240.1 888.0,240.3 892.0,240.4 896.0,240.5 900.0,240.6 904.0,240.7 908.0,240.9 912.0,241.0 916.0,241.1 920.0,241.2 924.0,241.3 928.0,241.4 932.0,241.5 936.0,241.6 940.0,241.8 944.0,241.9 948.0,242.0 952.0,242.1 956.0,242.1 960.0,242.2 964.0,242.3 968.0,242.4 972.0,242.5 976.0,242.6 980.0,242.7 984.0,242.8 988.0,242.9 992.0,243.0 996.0,243.0 1000.0,243.1 1004.0,243.2 1008.0,243.3 1012.0,243.4 1016.0,243.4 1020.0,243.5 1024.0,243.6 1028.0,243.7 1032.0,243.7 1036.0,243.8 1040.0,243.9 1044.0,243.9 1048.0,244.0 1052.0,244.1 1056.0,244.1 1060.0,244.2"/>
      <line className="tick" x1="340" y1="148" x2="340" y2="248"/>
      <line className="tick" x1="268" y1="148" x2="268" y2="248"/>
      <circle className="dot-a" cx="340" cy="148" r="4"/>
      <circle className="dot-g" cx="268" cy="148" r="4"/>
      <rect className="chip" x="352" y="86" width="290" height="20"/>
      <text className="tiny a-t" x="360" y="100">MODEL TRAINING — 30-DAY HALF-LIFE</text>
      <rect className="chip" x="352" y="176" width="330" height="20"/>
      <text className="tiny gold-t" x="360" y="190">EMPIRICAL WIN-RATE TABLES — 21-DAY HALF-LIFE</text>
    </svg>
  );
}

export function TrainLoop() {
  return (
    <svg className="d" viewBox="0 0 1120 292" role="img" aria-label="The training loop with two validation heads: a fixed masked copy drives early stopping, while unmasked full comps drive the publish gate.">
      <defs><marker id="m6" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#333a46"/></marker></defs>
      <rect className="box" x="30" y="40" width="240" height="44"/><text x="150" y="67" textAnchor="middle">train split — 85%</text>
      <path className="wire" d="M270,62 L326,62" markerEnd="url(#m6)"/>
      <rect className="box-k" x="330" y="40" width="240" height="44"/><text x="450" y="67" textAnchor="middle">re-mask (fresh RNG)</text>
      <path className="wire" d="M570,62 L626,62" markerEnd="url(#m6)"/>
      <rect className="box-k" x="630" y="40" width="240" height="44"/><text x="750" y="67" textAnchor="middle">AdamW · lr 1e-3 · b 256</text>
      <path className="wire" d="M750,84 L750,116 L450,116 L450,88" markerEnd="url(#m6)"/>
      <text className="tiny" x="600" y="134" textAnchor="middle">UP TO 40 EPOCHS</text>
      <path className="wire" d="M870,62 L916,62" markerEnd="url(#m6)"/>
      <rect className="box" x="920" y="40" width="170" height="44"/><text x="1005" y="67" textAnchor="middle">checkpoint</text>

      <path className="wire" d="M1005,84 L1005,152"/>
      <path className="wire" d="M1005,152 L300,152 L300,182" markerEnd="url(#m6)"/>
      <path className="wire" d="M1005,152 L790,152 L790,182" markerEnd="url(#m6)"/>

      <rect className="box-a" x="150" y="182" width="300" height="52"/>
      <text x="300" y="204" textAnchor="middle">val · FIXED masked copy</text>
      <text className="lbl" x="300" y="224" textAnchor="middle">early stop — patience 6</text>
      <rect className="box-b" x="640" y="182" width="300" height="52"/>
      <text x="790" y="204" textAnchor="middle">val · UNMASKED full 3v3</text>
      <text className="lbl" x="790" y="224" textAnchor="middle">gate — delta log-loss ≤ 0.0035</text>

      <text className="cap" x="300" y="258" textAnchor="middle">picks which epoch to keep</text>
      <text className="cap" x="790" y="258" textAnchor="middle">decides whether anything is written at all</text>
    </svg>
  );
}

export function EdgeCalibration() {
  return (
    <svg className="d" viewBox="0 0 1120 348" role="img" aria-label="As more of the draft is known, the model claims a larger edge while its calibration error holds between 0.011 and 0.014.">
      <line className="ax" x1="120" y1="280" x2="1060" y2="280"/>
      <line className="ax" x1="120" y1="60" x2="120" y2="280"/>
      <line className="grid" x1="120" y1="225" x2="1060" y2="225"/>
      <line className="grid" x1="120" y1="170" x2="1060" y2="170"/>
      <line className="grid" x1="120" y1="115" x2="1060" y2="115"/>
      <line className="grid" x1="120" y1="60" x2="1060" y2="60"/>
      <text className="tiny" x="112" y="284" textAnchor="end">0</text>
      <text className="tiny" x="112" y="229" textAnchor="end">.025</text>
      <text className="tiny" x="112" y="174" textAnchor="end">.050</text>
      <text className="tiny" x="112" y="119" textAnchor="end">.075</text>
      <text className="tiny" x="112" y="64" textAnchor="end">.100</text>

      <rect className="bar" x="160.3" y="204.9" width="76" height="75.1"/>
      <rect className="bar" x="317.0" y="171.0" width="76" height="109.0"/>
      <rect className="bar" x="473.7" y="139.1" width="76" height="140.9"/>
      <rect className="bar" x="630.3" y="111.3" width="76" height="168.7"/>
      <rect className="bar" x="787.0" y="86.3" width="76" height="193.7"/>
      <rect className="bar bar-hi" x="943.7" y="62.1" width="76" height="217.9"/>

      <text className="barval" x="198.3" y="197" textAnchor="middle">.034</text>
      <text className="barval" x="355.0" y="163" textAnchor="middle">.050</text>
      <text className="barval" x="511.7" y="131" textAnchor="middle">.064</text>
      <text className="barval" x="668.3" y="103" textAnchor="middle">.077</text>
      <text className="barval" x="825.0" y="78" textAnchor="middle">.088</text>
      <text className="barval" x="981.7" y="54" textAnchor="middle">.099</text>

      <polyline className="ece" points="198.3,252.0 355.0,250.2 511.7,252.5 668.3,251.7 825.0,254.9 981.7,254.9"/>
      <circle className="dot-g" cx="198.3" cy="252.0" r="4"/><circle className="dot-g" cx="355.0" cy="250.2" r="4"/>
      <circle className="dot-g" cx="511.7" cy="252.5" r="4"/><circle className="dot-g" cx="668.3" cy="251.7" r="4"/>
      <circle className="dot-g" cx="825.0" cy="254.9" r="4"/><circle className="dot-g" cx="981.7" cy="254.9" r="4"/>
      {/* Legend sits below the state ticks: at x=600 it would otherwise blank out the 2v2 and
          3v2 labels, since .chip paints an opaque background. */}
      <rect className="chip" x="596" y="316" width="342" height="20"/>
      <text className="tiny green-t" x="608" y="330">CALIBRATION ERROR — 0.011 TO 0.014 ACROSS STATES</text>

      <text className="tiny" x="198.3" y="300" textAnchor="middle">1v0</text>
      <text className="tiny" x="355.0" y="300" textAnchor="middle">1v1</text>
      <text className="tiny" x="511.7" y="300" textAnchor="middle">2v1</text>
      <text className="tiny" x="668.3" y="300" textAnchor="middle">2v2</text>
      <text className="tiny" x="825.0" y="300" textAnchor="middle">3v2</text>
      <text className="tiny" x="981.7" y="300" textAnchor="middle">3v3</text>
      <text className="lbl" x="120" y="330">how much of the draft is known —&gt;</text>
    </svg>
  );
}

/* ---------------------------------------------------------------------------
   Signal shares: what fraction of the fused score each signal actually owns at
   four points in a draft. Derived, not drawn — the numbers below mirror
   DEFAULT_WEIGHTS in backend/bsdraft/engine/scoring.py, and the bar geometry
   falls out of the same renormalization the engine does (inactive signals are
   dropped from the average entirely, so the divisor is the sum of the *active*
   weights, not 1).
   --------------------------------------------------------------------------- */

export const WEIGHTS = {
  model: 0.40, map: 0.25, counter: 0.20, synergy: 0.05, role: 0.10, mastery: 0.10, personal: 0.08,
} as const;

type Signal = keyof typeof WEIGHTS;

const FILL: Record<Signal, string> = {
  model: "sg-model", map: "sg-map", counter: "sg-cnt", synergy: "sg-syn",
  role: "sg-role", mastery: "sg-mas", personal: "sg-per",
};
// The four signals with weights fitted on held-out matches render saturated; the three
// hand-set heuristics render grey. That contrast is the point of the figure.
const HEURISTIC: Signal[] = ["role", "mastery", "personal"];

// `limit` marks a row drawn at an idealization: the engine scales the personal weight by
// confidence (games/(games+20)), so the personalized row is the conf -> 1 ceiling, and a real
// board's personal slice is always thinner than what is drawn.
const STATES: { label: string; limit?: string; on: Signal[] }[] = [
  { label: "First pick", on: ["model", "map", "role"] },
  { label: "Allies picked", on: ["model", "map", "synergy", "role"] },
  { label: "Both sides revealed", on: ["model", "map", "counter", "synergy", "role"] },
  { label: "Personalized", limit: "CONF \u2192 1", on: ["model", "map", "counter", "synergy", "role", "mastery", "personal"] },
];

const X0 = 300, BAR_W = 760, ROW_H = 36, ROW_GAP = 66, TOP = 60, LABEL_MIN = 48;

export function SignalShares() {
  const rows = STATES.map((st, i) => {
    const divisor = st.on.reduce((a, k) => a + WEIGHTS[k], 0);
    let x = X0;
    const segs = st.on.map((k) => {
      const share = WEIGHTS[k] / divisor;
      const w = share * BAR_W;
      const seg = { k, x, w, share };
      x += w;
      return seg;
    });
    return { ...st, divisor, segs, y: TOP + i * ROW_GAP };
  });
  const last = rows[rows.length - 1];
  const modelFirst = rows[0].segs[0].share, modelLast = last.segs[0].share;

  return (
    <svg className="d" viewBox={`0 0 1120 ${TOP + STATES.length * ROW_GAP + 26}`} role="img"
      aria-label={`Stacked bars showing each signal's effective share of the fused score at four draft states; the model's share falls from ${(modelFirst * 100).toFixed(0)} percent at first pick to ${(modelLast * 100).toFixed(0)} percent on a personalized full board.`}>
      {rows.map((r) => (
        <g key={r.label}>
          <text className="lbl" x={X0 - 14} y={r.y + 12} textAnchor="end">{r.label}</text>
          <text className="tiny" x={X0 - 14} y={r.y + 28} textAnchor="end">
            DIVISOR {r.divisor.toFixed(2)}{r.limit ? ` \u00b7 ${r.limit}` : ""}
          </text>
          {r.segs.map((s) => (
            <g key={s.k}>
              <rect className={`sg ${FILL[s.k]}`} x={s.x} y={r.y} width={s.w} height={ROW_H} />
              {s.w >= LABEL_MIN && (
                <text className={`sgv${s.k === "mastery" ? " dark" : ""}`} x={s.x + s.w / 2} y={r.y + 23}
                  textAnchor="middle">{(s.share * 100).toFixed(1)}%</text>
              )}
            </g>
          ))}
        </g>
      ))}
      <g className="lg">
        {(Object.keys(WEIGHTS) as Signal[]).map((k, i) => (
          <g key={k}>
            <rect className={`sg ${FILL[k]}`} x={X0 + i * 112} y={TOP + STATES.length * ROW_GAP + 2} width="12" height="12" />
            <text className="tiny" x={X0 + i * 112 + 18} y={TOP + STATES.length * ROW_GAP + 12}>
              {k.toUpperCase()}{HEURISTIC.includes(k) ? " *" : ""}
            </text>
          </g>
        ))}
      </g>
    </svg>
  );
}

export function StackingSplit() {
  return (
    <svg className="d" viewBox="0 0 1120 250" role="img" aria-label="A stacking regression's split between the neural net and empirical statistics reversed from 31/69 to 78/22 as the training set grew from 40 thousand to 995 thousand matches.">
      <text className="lbl" x="286" y="82" textAnchor="end">JUN 2026</text>
      <text className="tiny" x="286" y="98" textAnchor="end">40,208 MATCHES</text>
      <rect className="sg sg-model" x="300" y="70" width="217" height="44"/><text className="sgv" x="408" y="97" textAnchor="middle">NET 31%</text>
      <rect className="sg sg-emp" x="517" y="70" width="483" height="44"/><text className="sgv dark" x="758" y="97" textAnchor="middle">COUNT TABLES 69%</text>
      <text className="tiny" x="1016" y="90">AUC</text><text className="barval" x="1016" y="107">.576</text>

      <text className="lbl" x="286" y="172" textAnchor="end">AUG 2026</text>
      <text className="tiny" x="286" y="188" textAnchor="end">995,135 MATCHES</text>
      <rect className="sg sg-model" x="300" y="160" width="546" height="44"/><text className="sgv" x="573" y="187" textAnchor="middle">NET 78%</text>
      <rect className="sg sg-emp" x="846" y="160" width="154" height="44"/><text className="sgv dark" x="923" y="187" textAnchor="middle">22%</text>
      <text className="tiny" x="1016" y="180">AUC</text><text className="barval" x="1016" y="197">.627</text>

      <text className="lbl" x="300" y="232">same harness · same held-out protocol · 25× the data · opposite conclusion</text>
    </svg>
  );
}
