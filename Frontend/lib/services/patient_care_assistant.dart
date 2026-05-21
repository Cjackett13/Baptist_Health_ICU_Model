import '../models/patient_prediction.dart';

/// Demo care assistant — answers from the patient's chart (no external API).
class PatientCareAssistant {
  const PatientCareAssistant._();

  static String reply(PatientRecord patient, String question) {
    final q = question.trim().toLowerCase();
    if (q.isEmpty) {
      return 'Ask me about your medications, estimated costs, or recommendations '
          'from your care team.';
    }

    if (_matches(q, ['medication', 'medicine', 'drug', 'prescri', 'taking', 'dose', 'dosage'])) {
      return _medicationsAnswer(patient);
    }
    if (_matches(q, ['price', 'cost', 'afford', 'pay', 'insurance', 'expensive', 'copay'])) {
      return _pricingAnswer(patient);
    }
    if (_matches(q, [
      'recommend',
      'improve',
      'health',
      'diet',
      'exercise',
      'lifestyle',
      'recover',
      'better',
      'tip',
      'advice',
    ])) {
      return _recommendationsAnswer(patient);
    }
    if (_matches(q, ['stay', 'hospital', 'discharge', 'leave', 'how long', 'los'])) {
      final days = patient.predictions.hospitalLosDays;
      return 'Your care team estimates about ${days.toStringAsFixed(1)} days in the '
          'hospital. This can change as you improve — ask your nurse or doctor for the '
          'latest plan.';
    }
    if (_matches(q, ['doctor', 'attending', 'physician', 'who is my'])) {
      final doc = patient.primaryDoctor ?? 'your attending physician';
      return 'Your listed care team lead is $doc at ${patient.roomNumber}. '
          'For clinical decisions, always follow what they tell you in person.';
    }
    if (_matches(q, ['diagnosis', 'condition', 'why am i', 'heart'])) {
      final dx = patient.diagnosis ?? patient.issue ?? 'your heart condition';
      return 'Your chart shows a principal concern of $dx. Your team is monitoring you '
          'closely in the ICU. This assistant cannot replace medical advice from them.';
    }

    return 'I can help with medications, typical medication cost questions, recovery '
        'recommendations on your chart, and estimated length of stay. '
        'For urgent symptoms, contact your nurse or call your care team right away.';
  }

  static bool _matches(String q, List<String> keys) =>
      keys.any((k) => q.contains(k));

  static String _medicationsAnswer(PatientRecord patient) {
    if (patient.medications.isEmpty) {
      return 'No medications are listed on your chart right now. Ask your nurse if '
          'something was started recently.';
    }
    final buf = StringBuffer('Here are the medications on your record:\n\n');
    for (final m in patient.medications) {
      buf.writeln('• ${m.name} — ${m.dosage} (${m.route})');
      if (m.isVasopressor) {
        buf.writeln('  This is a heart-support IV medicine managed by your ICU team.');
      }
    }
    buf.write('\nDo not change doses on your own. Talk to your nurse or doctor before '
        'starting or stopping anything.');
    return buf.toString();
  }

  static String _pricingAnswer(PatientRecord patient) {
    if (patient.medications.isEmpty) {
      return 'I do not see medications on your chart to price. Many ICU medicines are '
          'billed as part of your hospital stay, not like a retail pharmacy pickup.';
    }
    final buf = StringBuffer(
      'Estimated costs below are for planning only — not a quote from Baptist Health. '
      'ICU IV medicines are usually part of your hospital bill.\n\n',
    );
    for (final m in patient.medications) {
      buf.writeln('• ${m.name}: ${_estimatePrice(m)}');
    }
    buf.write('\nFor exact charges, ask financial counseling or your insurer. '
        'Retail prices apply only if you take a medicine at home after discharge.');
    return buf.toString();
  }

  static String _estimatePrice(PatientMedication m) {
    final code = m.code.toLowerCase();
    if (m.isVasopressor) {
      return 'Given in ICU — typically covered under inpatient billing, not a retail fill.';
    }
    if (code.contains('furosemide')) {
      return 'Generic tablet at home often ~\$4–\$20/month; IV in hospital is part of stay charges.';
    }
    if (code.contains('metoprolol') || code.contains('carvedilol')) {
      return 'Generic beta-blocker often ~\$4–\$30/month with insurance at a pharmacy.';
    }
    if (code.contains('aspirin')) {
      return 'Low-cost OTC — often under \$10/month.';
    }
    if (code.contains('heparin')) {
      return 'Hospital inpatient therapy — billed with your ICU stay.';
    }
    return 'Ask your pharmacy or insurer; inpatient IV doses differ from home retail prices.';
  }

  static String _recommendationsAnswer(PatientRecord patient) {
    if (patient.recommendations.isEmpty) {
      return 'No personalized recommendations are on file yet. General tips: take medicines '
          'exactly as prescribed, report chest pain or shortness of breath promptly, '
          'and follow activity limits from your nurse.';
    }
    final buf = StringBuffer('Recommendations from your care plan:\n\n');
    for (final r in patient.recommendations) {
      buf.writeln('• ${r.title}');
      buf.writeln('  ${r.description}');
      buf.writeln('  Goal: ${r.impact}\n');
    }
    buf.write('These are meant to support recovery — follow your team if anything conflicts.');
    return buf.toString();
  }
}
