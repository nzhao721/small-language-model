import {
  BarChart,
  Callout,
  Card,
  CardBody,
  CardHeader,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  PieChart,
  Stack,
  Stat,
  Table,
  Text,
} from "cursor/canvas";

const groupNames = [
  "HW orig.",
  "Clean orig.",
  "Camera orig.",
  "HW corrupt",
  "Clean corrupt",
  "Camera corrupt",
  "Manual HW",
  "Manual digital",
];

const groupAccuracy = [98.49, 100, 98.5, 98.81, 100, 99.57, 93.16, 98.69];
const groupExactMatch = [80, 100, 90, 80, 100, 90, 15, 65];

const groupRows = [
  ["Handwritten original", "10", "98.49%", "80%", "4"],
  ["Clean original", "10", "100.00%", "100%", "0"],
  ["Camera-distorted original", "10", "98.50%", "90%", "3"],
  ["Handwritten corrupt", "10", "98.81%", "80%", "3"],
  ["Clean corrupt", "10", "100.00%", "100%", "0"],
  ["Camera-distorted corrupt", "10", "99.57%", "90%", "1"],
  ["Manual handwritten", "20", "93.16%", "15%", "37"],
  ["Manual digital", "20", "98.69%", "65%", "7"],
];

const hardestRows = [
  ["gs8", "Manual handwritten", "85.19%", "4 / 27", "Eighth notes read as quarters"],
  ["gs12", "Manual handwritten", "85.71%", "4 / 28", "Header + pitch/duration sequence"],
  ["220019566-1_1_1", "Camera-distorted original", "85.00%", "3 / 20", "Dropped fermata note and token"],
  ["gs3", "Manual handwritten", "88.89%", "3 / 27", "Pitch substitutions in bass clef"],
  ["gs9", "Manual handwritten", "88.46%", "3 / 26", "Pitch sequence after otherwise correct header"],
  ["gs0", "Manual handwritten", "93.10%", "2 / 29", "Two melodic pitch substitutions"],
  ["gs4", "Manual handwritten", "92.31%", "2 / 26", "Duration substitutions"],
  ["gs6", "Manual handwritten", "91.67%", "2 / 24", "Missing key signature + pitch substitution"],
];

function App() {
  return (
    <Stack gap={20} style={{ padding: 24, maxWidth: 1180, margin: "0 auto" }}>
      <Stack gap={6}>
        <H1>Qwen2-VL-2B final: remaining errors are concentrated in real handwriting</H1>
        <Text tone="secondary">
          Checkpoint-9000 · 2.07 epochs · 100-image golden set · source:
          <Text as="span" weight="semibold"> eval_qwen2-vl-2b.json</Text>
        </Text>
      </Stack>

      <Grid columns={5} gap={16}>
        <Stat value="97.91%" label="Mean token accuracy" tone="success" />
        <Stat value="70 / 100" label="Exact matches" tone="success" />
        <Stat value="55" label="Total token edits" tone="warning" />
        <Stat value="37 / 55" label="Edits in manual handwriting" tone="danger" />
        <Stat value="0" label="Inference failures" tone="success" />
      </Grid>

      <Callout tone="warning" title="Main conclusion">
        The model has effectively saturated the clean synthetic slices, while manual handwritten
        notation remains the limiting domain. Manual handwriting represents 20% of the set but
        67% of all remaining token edits and 17 of the 30 non-exact samples.
      </Callout>

      <Grid columns="minmax(0, 1.55fr) minmax(300px, 0.9fr)" gap={20} align="start">
        <Stack gap={8}>
          <H2>Token accuracy by evaluation group</H2>
          <BarChart
            categories={groupNames}
            series={[{ name: "Token accuracy", data: groupAccuracy, tone: "success" }]}
            height={360}
            yMin={90}
            yMax={100}
            valueSuffix="%"
            showValues
            referenceLines={[{ value: 95, label: "95% reference", tone: "warning" }]}
          />
          <Text size="small" tone="tertiary">
            X-axis: golden-set evaluation group · Y-axis: mean token accuracy (%) · Source:
            checkpoint-9000 golden evaluation · synthetic groups n=10; manual groups n=20.
          </Text>
        </Stack>

        <Card size="lg">
          <CardHeader>Where the 55 edits occur</CardHeader>
          <CardBody>
            <Stack gap={12}>
              <PieChart
                donut
                size={230}
                data={[
                  { label: "Manual handwritten", value: 37, tone: "danger" },
                  { label: "Manual digital", value: 7, tone: "warning" },
                  { label: "Synthetic handwritten", value: 7, tone: "info" },
                  { label: "Synthetic camera", value: 4, tone: "neutral" },
                ]}
              />
              <Text size="small" tone="tertiary">
                Token edits after the evaluation’s key-signature, common/cut-time, and trailing
                barline normalization. Clean synthetic groups contribute zero edits.
              </Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <Stack gap={8}>
        <H2>Accuracy and exact-match detail</H2>
        <Table
          headers={["Group", "n", "Token accuracy", "Exact match", "Token edits"]}
          rows={groupRows}
          columnAlign={["left", "right", "right", "right", "right"]}
          rowTone={[
            "success",
            "success",
            "success",
            "success",
            "success",
            "success",
            "danger",
            "warning",
          ]}
          striped
        />
      </Stack>

      <Divider />

      <Grid columns="minmax(0, 1.3fr) minmax(300px, 0.9fr)" gap={20} align="start">
        <Stack gap={8}>
          <H2>Exact-match rate exposes the remaining reliability gap</H2>
          <BarChart
            categories={groupNames}
            series={[{ name: "Exact match", data: groupExactMatch, tone: "warning" }]}
            height={340}
            yMin={0}
            yMax={100}
            valueSuffix="%"
            showValues
          />
          <Text size="small" tone="tertiary">
            X-axis: golden-set evaluation group · Y-axis: samples with zero normalized token
            edits (%) · Source: checkpoint-9000 golden evaluation.
          </Text>
        </Stack>

        <Card size="lg">
          <CardHeader>Normalized token edits by error type</CardHeader>
          <CardBody>
            <Stack gap={12}>
              <PieChart
                donut
                size={250}
                data={[
                  { label: "Duration / articulation", value: 23, tone: "warning" },
                  { label: "Pitch only", value: 18, tone: "danger" },
                  { label: "Key signature", value: 8, tone: "info" },
                  { label: "Pitch + duration", value: 1, tone: "danger" },
                  { label: "Time signature", value: 1, tone: "info" },
                  { label: "Barline", value: 1, tone: "neutral" },
                  { label: "Insertion", value: 1, tone: "neutral" },
                  { label: "Deletion", value: 2, tone: "neutral" },
                ]}
              />
              <Text size="small" tone="tertiary">
                Source: normalized, minimum-edit token alignment for all 55 edits. Duration /
                articulation includes note or rest duration changes and fermata loss. There were
                zero clef or uncategorized-token edits, so those categories are omitted.
              </Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <Grid columns="minmax(0, 1.3fr) minmax(300px, 0.9fr)" gap={20} align="start">
        <Callout tone="info" title="What the remaining mistakes look like">
          Most misses are local, not full-sequence failures: a note pitch or duration
          substitution, omitted key signature, or lost fermata. The model never failed
          inference and most non-exact sequences remain above 85% token accuracy.
        </Callout>
        <Card>
          <CardHeader>Observed error patterns</CardHeader>
          <CardBody>
            <Stack gap={10}>
              <Text>
                <Text as="span" weight="semibold">Rhythm:</Text> handwritten eighth notes are
                occasionally read as quarters; fermata modifiers can be omitted.
              </Text>
              <Text>
                <Text as="span" weight="semibold">Pitch:</Text> residual substitutions cluster
                in handwritten, especially bass-clef material and later-note sequences.
              </Text>
              <Text>
                <Text as="span" weight="semibold">Headers:</Text> some key signatures are
                wrong or absent. Relative major/minor aliases and C/C/ time aliases are already
                normalized by scoring, so those equivalent visual spellings do not inflate edits.
              </Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <Stack gap={8}>
        <H2>Hardest remaining examples</H2>
        <Table
          headers={["Sample", "Group", "Accuracy", "Edits / GT tokens", "Primary residual error"]}
          rows={hardestRows}
          columnAlign={["left", "left", "right", "right", "left"]}
          rowTone={["danger", "danger", "danger", "warning", "warning", "warning", "warning", "warning"]}
          striped
        />
        <Text size="small" tone="tertiary">
          Source: per-sample normalized Levenshtein results. “Primary residual error” is a
          qualitative reading of the ground-truth / prediction differences, not a model confidence score.
        </Text>
      </Stack>

      <H2>Most useful next iteration</H2>
      <Grid columns="1fr 1fr" gap={16}>
        <Card>
          <CardHeader trailing="Priority 1">Add non-golden manual-style adaptation data</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <H3>Target the actual weak domain</H3>
              <Text>
                Add held-out-source training examples with uneven staff lines, faint strokes,
                skew, bass clef, and imperfect handwritten noteheads. Do not train on the 20
                manual golden images.
              </Text>
            </Stack>
          </CardBody>
        </Card>
        <Card>
          <CardHeader trailing="Priority 2">Reweight for hard visual tokens</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <H3>Shift away from saturated clean data</H3>
              <Text>
                Clean synthetic performance is already perfect. Favor manual-like handwriting,
                fermatas, short-duration notes, key signatures, and bass-clef sequences in a
                short, low-learning-rate adaptation stage.
              </Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <Callout tone="success" title="Decision">
        Preserve checkpoint-9000 as the release candidate. Scaling or simply training longer is
        unlikely to materially improve the remaining errors without more representative manual
        handwriting data; the evidence points to domain coverage rather than a broad OMR failure.
      </Callout>
    </Stack>
  );
}

export default App;
