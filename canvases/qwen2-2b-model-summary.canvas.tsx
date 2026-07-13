import {
  BarChart,
  Callout,
  Card,
  CardBody,
  CardHeader,
  Grid,
  H1,
  H2,
  Stack,
  Text,
  useHostTheme,
} from "cursor/canvas";

type ModelSlice = {
  model: string;
  overall: string;
  original: string;
  corrupt: string;
  digital: string;
  camera: string;
  handwritten: string;
};

const groups = [
  "Overall",
  "Original",
  "Mistakes",
  "Digital",
  "Camera",
  "Handwritten",
];

// Each cell is "mean token accuracy % · exact match %".
const fineTunedRows: ModelSlice[] = [
  {
    model: "Florence-2-base fine-tuned",
    overall: "93.29 · 38.0",
    original: "92.13 · 34.3",
    corrupt: "95.98 · 46.7",
    digital: "96.91 · 60.0",
    camera: "96.00 · 55.0",
    handwritten: "88.31 · 7.5",
  },
  {
    model: "Florence-2-large fine-tuned",
    overall: "94.79 · 45.0",
    original: "94.42 · 40.0",
    corrupt: "95.66 · 56.7",
    digital: "98.12 · 65.0",
    camera: "95.57 · 55.0",
    handwritten: "91.08 · 20.0",
  },
  {
    model: "Qwen2-VL-2B fine-tuned",
    overall: "97.91 · 70.0",
    original: "97.24 · 61.4",
    corrupt: "99.31 · 90.0",
    digital: "99.35 · 82.5",
    camera: "99.03 · 90.0",
    handwritten: "95.91 · 47.5",
  },
];

const fineRows = fineTunedRows.map((row) => [
  row.model,
  row.overall,
  row.original,
  row.corrupt,
  row.digital,
  row.camera,
  row.handwritten,
]);

// Each cell is "mean token accuracy % · exact match %".
const frontierRows: ModelSlice[] = [
  {
    model: "Opus 4.8 Thinking Max",
    overall: "48.28 · 5.0",
    original: "54.52 · 7.1",
    corrupt: "33.71 · 0.0",
    digital: "80.24 · 12.5",
    camera: "36.73 · 0.0",
    handwritten: "22.09 · 0.0",
  },
  {
    model: "Gemini 3.1 Pro",
    overall: "14.85 · 0.0",
    original: "16.77 · 0.0",
    corrupt: "10.36 · 0.0",
    digital: "20.38 · 0.0",
    camera: "3.30 · 0.0",
    handwritten: "15.09 · 0.0",
  },
  {
    model: "Composer 2.5",
    overall: "31.43 · 0.0",
    original: "30.12 · 0.0",
    corrupt: "34.47 · 0.0",
    digital: "35.40 · 0.0",
    camera: "28.81 · 0.0",
    handwritten: "28.76 · 0.0",
  },
  {
    model: "Grok 4.5 high",
    overall: "38.12 · 0.0",
    original: "40.97 · 0.0",
    corrupt: "31.47 · 0.0",
    digital: "52.17 · 0.0",
    camera: "27.10 · 0.0",
    handwritten: "29.58 · 0.0",
  },
  {
    model: "GPT-5.6 Sol High",
    overall: "48.50 · 8.0",
    original: "50.84 · 11.4",
    corrupt: "43.06 · 0.0",
    digital: "79.98 · 20.0",
    camera: "32.20 · 0.0",
    handwritten: "25.18 · 0.0",
  },
];

const frontierTableRows = frontierRows.map((row) => [
  row.model,
  row.overall,
  row.original,
  row.corrupt,
  row.digital,
  row.camera,
  row.handwritten,
]);

// Ordered by overall token accuracy, descending.
const allModelRows = [
  fineRows[2], // Qwen2-VL-2B fine-tuned: 97.91%
  fineRows[1], // Florence-2-large: 94.79%
  fineRows[0], // Florence-2-base: 93.29%
  frontierTableRows[4], // GPT-5.6 Sol High: 48.50%
  frontierTableRows[0], // Opus 4.8 Thinking Max: 48.28%
  frontierTableRows[3], // Grok 4.5 high: 38.12%
  frontierTableRows[2], // Composer 2.5: 31.43%
  frontierTableRows[1], // Gemini 3.1 Pro: 14.85%
];

function ModelComparisonTable() {
  const theme = useHostTheme();
  const columns = "minmax(220px, 1.45fr) repeat(6, minmax(108px, 1fr))";

  return (
    <div
      style={{
        overflowX: "auto",
        border: `1px solid ${theme.stroke.secondary}`,
        borderRadius: 6,
      }}
    >
      <div style={{ minWidth: 920 }}>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: columns,
            background: theme.fill.primary,
            borderBottom: `1px solid ${theme.stroke.secondary}`,
          }}
        >
          {["Model", ...groups].map((header, index) => (
            <div
              key={header}
              style={{
                padding: "8px 10px",
                textAlign: index === 0 ? "left" : "right",
              }}
            >
              <Text size="small" weight="semibold">
                {header}
              </Text>
            </div>
          ))}
        </div>

        {allModelRows.map((row, rowIndex) => {
          const fineTuned = rowIndex < 3;
          return (
            <div
              key={String(row[0])}
              style={{
                display: "grid",
                gridTemplateColumns: columns,
                background: fineTuned ? theme.fill.secondary : theme.fill.tertiary,
                borderTop:
                  rowIndex === 0 ? undefined : `1px solid ${theme.stroke.tertiary}`,
              }}
            >
              {row.map((cell, index) => (
                <div
                  key={`${String(row[0])}-${index}`}
                  style={{
                    padding: "8px 10px",
                    textAlign: index === 0 ? "left" : "right",
                    whiteSpace: "nowrap",
                  }}
                >
                  <Text size="small" weight={index === 0 ? "semibold" : undefined}>
                    {cell}
                  </Text>
                </div>
              ))}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function FrontierZeroLabels({
  gptZeroIndices,
  opusZeroIndices,
}: {
  gptZeroIndices: number[];
  opusZeroIndices: number[];
}) {
  return (
    <div
      style={{
        position: "absolute",
        left: 54,
        right: 18,
        bottom: 58,
        pointerEvents: "none",
      }}
    >
      <Grid columns={6} gap={0}>
        {groups.map((group, index) => (
          <div key={group}>
            <Grid columns={3} gap={0}>
              <div />
              <div>
                {gptZeroIndices.includes(index) && (
                  <Text
                    size="small"
                    weight="semibold"
                    style={{ textAlign: "center", whiteSpace: "nowrap" }}
                  >
                    0%
                  </Text>
                )}
              </div>
              <div>
                {opusZeroIndices.includes(index) && (
                  <Text
                    size="small"
                    weight="semibold"
                    style={{ textAlign: "center", whiteSpace: "nowrap" }}
                  >
                    0%
                  </Text>
                )}
              </div>
            </Grid>
          </div>
        ))}
      </Grid>
    </div>
  );
}

export default function Qwen2BModelSummary() {
  return (
    <Stack gap={22} style={{ padding: 24, maxWidth: 1320, margin: "0 auto" }}>
      <Stack gap={6}>
        <H1>Qwen2-VL-2B fine-tuned benchmark summary</H1>
        <Text tone="secondary">
          100-image golden-set comparison of the three fine-tuned models and frontier
          models.
        </Text>
      </Stack>

      <Grid columns={2} gap={20}>
        <Stack gap={4} style={{ textAlign: "center", alignItems: "center" }}>
          <div style={{ fontSize: 24, lineHeight: "30px", fontWeight: 590 }}>
            97.91%
          </div>
          <Text weight="semibold" size="small">
            Qwen2-VL-2B fine-tuned token accuracy
          </Text>
          <Text tone="secondary" size="small">
            +49.41 pp vs GPT-5.6 Sol High
          </Text>
        </Stack>
        <Stack gap={4} style={{ textAlign: "center", alignItems: "center" }}>
          <div style={{ fontSize: 24, lineHeight: "30px", fontWeight: 590 }}>
            70%
          </div>
          <Text weight="semibold" size="small">
            Qwen2-VL-2B fine-tuned exact match
          </Text>
          <Text tone="secondary" size="small">
            +62 pp vs GPT-5.6 Sol High
          </Text>
        </Stack>
      </Grid>

      <Stack gap={18}>
        <Card size="lg">
          <CardHeader>Token accuracy by evaluation slice</CardHeader>
          <CardBody>
            <Stack gap={10}>
              <div style={{ position: "relative" }}>
                <BarChart
                  categories={groups}
                  series={[
                    {
                      name: "Qwen2-VL-2B fine-tuned",
                      data: [97.91, 97.24, 99.31, 99.35, 99.03, 95.91],
                      tone: "success",
                    },
                    {
                      name: "GPT-5.6 Sol High",
                      data: [48.5, 50.84, 43.06, 79.98, 32.2, 25.18],
                      tone: "info",
                    },
                    {
                      name: "Opus 4.8 Thinking Max",
                      data: [48.28, 54.52, 33.71, 80.24, 36.73, 22.09],
                      tone: "neutral",
                    },
                  ]}
                  height={330}
                  yMin={0}
                  yMax={100}
                  valueSuffix="%"
                  showValues
                />
              </div>
              <Text tone="tertiary" size="small">
                X-axis: evaluation slice. Y-axis: mean token accuracy (%). Best
                frontier model is selected by overall accuracy.
              </Text>
            </Stack>
          </CardBody>
        </Card>

        <Card size="lg">
          <CardHeader>Exact match by evaluation slice</CardHeader>
          <CardBody>
            <Stack gap={10}>
              <div style={{ position: "relative" }}>
                <BarChart
                  categories={groups}
                  series={[
                    {
                      name: "Qwen2-VL-2B fine-tuned",
                      data: [70, 61.4, 90, 82.5, 90, 47.5],
                      tone: "success",
                    },
                    {
                      name: "GPT-5.6 Sol High",
                      data: [8, 11.4, 0, 20, 0, 0],
                      tone: "info",
                    },
                    {
                      name: "Opus 4.8 Thinking Max",
                      data: [5, 7.1, 0, 12.5, 0, 0],
                      tone: "neutral",
                    },
                  ]}
                  height={330}
                  yMin={0}
                  yMax={100}
                  valueSuffix="%"
                  showValues
                />
                <FrontierZeroLabels
                  gptZeroIndices={[2, 4, 5]}
                  opusZeroIndices={[2, 4, 5]}
                />
              </div>
              <Text tone="tertiary" size="small">
                X-axis: evaluation slice. Y-axis: samples with zero token edits (%).
                Frontier scores use the same golden set.
              </Text>
            </Stack>
          </CardBody>
        </Card>
      </Stack>

      <Text tone="tertiary" size="small">
        Zero-shot Qwen2-VL-2B, Florence-2-base, and Florence-2-large produced either
        an empty string or a long nonsensical token sequence, so they are not plotted.
      </Text>

      <Stack gap={8}>
        <H2>Fine-tuned and frontier model comparison</H2>
        <Text tone="secondary" size="small">
          Each value is token accuracy % · exact match %. Exact match requires zero
          token edits after key/time-signature normalization and trailing-barline
          forgiveness. Model rows are sorted by overall token accuracy, descending.
        </Text>
        <ModelComparisonTable />
      </Stack>

      <Callout tone="info" title="Evaluation slices and scoring">
        Overall: Average across the entire set (100). Original: excerpts of real,
        original music (70). Mistakes: excerpts with simulated student mistakes
        injected randomly (30). Digital: clean, digitally rendered sheet music (40).
        Camera: photos or scans of printed music excerpts (20). Handwritten: photos
        or scans of handwritten music (40).
        <br />
        <br />
        The category counts do not add to 100 because every excerpt belongs to two
        groups: one of Original or Mistakes, and one of Digital, Camera, or
        Handwritten.
        <br />
        <br />
        Token accuracy uses token-level Levenshtein edit distance: each insert,
        deletion, or replacement is one edit; accuracy =
        max(0, 1 − edits / ground-truth-token-count) × 100. Exact match is the
        share of samples with zero edits. Relative major/minor keys, 4/4 ↔ C, and
        2/2 ↔ C/ are normalized as equivalent; one trailing barline on either side is
        forgiven.
      </Callout>

      <Text tone="tertiary" size="small">
        Sources: `scores_100.json` (frontier models), `scores_vlm_shortlist.md`
        (zero-shot base models), and the three fine-tuned evaluation JSON files.
        All figures are derived from the same 100-image golden set.
      </Text>
    </Stack>
  );
}
