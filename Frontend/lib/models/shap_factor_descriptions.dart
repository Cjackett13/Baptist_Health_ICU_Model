/// One-sentence clinical explanations for SHAP factor rows.
String describeShapFactor(String featureLabel, {required bool increasesRisk}) {
  final label = featureLabel.toLowerCase();
  final verb = increasesRisk ? 'raises' : 'lowers';
  final adj = increasesRisk ? 'higher' : 'lower';

  if (label.contains('scai')) {
    return increasesRisk
        ? 'A more advanced shock stage suggests greater near-term escalation risk.'
        : 'A less severe shock stage suggests lower near-term escalation risk.';
  }
  if (label.contains('shock burden')) {
    return increasesRisk
        ? 'Greater overall shock burden is pushing the estimate toward escalation.'
        : 'Lower shock burden is pulling the estimate away from escalation.';
  }
  if (label.contains('vasoactive') ||
      label.contains('inotrope') ||
      label.contains('vis')) {
    return increasesRisk
        ? '$adj vasoactive support intensity $verb the modeled escalation risk.'
        : '$adj vasoactive support intensity $verb the modeled escalation risk.';
  }
  if (label.contains('vasopressor')) {
    return increasesRisk
        ? 'More active vasopressor use $verb the chance the team will escalate support.'
        : 'Less vasopressor use $verb the chance the team will escalate support.';
  }
  if (label.contains('mean arterial') || label.contains('map')) {
    return increasesRisk
        ? 'Blood pressure patterns consistent with instability $verb escalation concern.'
        : 'Stronger blood pressure support $verb escalation concern.';
  }
  if (label.contains('lactate')) {
    return increasesRisk
        ? '$adj lactate $verb concern for worsening perfusion and escalation.'
        : '$adj lactate $verb concern for worsening perfusion and escalation.';
  }
  if (label.contains('oxygen') || label.contains('spo2')) {
    return increasesRisk
        ? 'Hypoxemia or borderline oxygenation $verb the escalation estimate.'
        : 'Better oxygenation $verb the escalation estimate.';
  }
  if (label.contains('heart rate')) {
    return increasesRisk
        ? 'Tachycardia or persistent elevated heart rate $verb escalation concern.'
        : 'More controlled heart rate $verb escalation concern.';
  }
  if (label.contains('creatinine') || label.contains('bun')) {
    return increasesRisk
        ? 'Renal strain on the profile $verb the modeled need for advanced support.'
        : 'More favorable renal markers $verb the modeled need for advanced support.';
  }
  if (label.contains('troponin') || label.contains('nt-probnp') || label.contains('nt probnp')) {
    return increasesRisk
        ? 'Cardiac injury or congestion signals $verb the escalation estimate.'
        : 'Milder cardiac biomarker burden $verb the escalation estimate.';
  }
  if (label.contains('hemoglobin') || label.contains('hgb')) {
    return increasesRisk
        ? 'Anemia or falling hemoglobin $verb concern for compensatory escalation.'
        : 'More stable hemoglobin $verb concern for compensatory escalation.';
  }
  if (label.contains('urine')) {
    return increasesRisk
        ? 'Low urine output $verb concern for cardiorenal shock and escalation.'
        : 'Adequate urine output $verb concern for cardiorenal shock and escalation.';
  }
  if (label.contains('hours on current')) {
    return increasesRisk
        ? 'Prolonged time at the current shock stage $verb urgency for escalation.'
        : 'Less time stuck at the current stage $verb urgency for escalation.';
  }

  return increasesRisk
      ? 'This pattern in the chart $verb the modeled escalation risk.'
      : 'This pattern in the chart $verb the modeled escalation risk.';
}
