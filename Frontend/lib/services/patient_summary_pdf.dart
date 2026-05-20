import 'package:pdf/pdf.dart';
import 'package:pdf/widgets.dart' as pw;
import 'package:printing/printing.dart';

import '../models/patient_prediction.dart';

/// Family-facing PDF — no mortality, MCS/ECMO, vitals, or SCAI (avoids undue anxiety).
class PatientSummaryPdf {
  /// [forFamily] when true (patient portal or share-with-family), omits clinical risk scores.
  static Future<void> share(
    PatientRecord patient, {
    bool forFamily = true,
  }) async {
    final doc = pw.Document();
    final updated = patient.predictions.lastUpdated;
    final dateStr =
        '${updated.month}/${updated.day}/${updated.year}';

    doc.addPage(
      pw.MultiPage(
        pageFormat: PdfPageFormat.letter,
        margin: const pw.EdgeInsets.all(40),
        build: (context) => forFamily
            ? _familyContent(patient, dateStr)
            : _clinicalContent(patient, dateStr),
      ),
    );

    final safeName =
        patient.name.replaceAll(RegExp(r'[^\w\s-]'), '').trim();
    await Printing.sharePdf(
      bytes: await doc.save(),
      filename: 'baptist_care_summary_${safeName.replaceAll(" ", "_")}.pdf',
    );
  }

  static List<pw.Widget> _familyContent(PatientRecord patient, String dateStr) {
    final p = patient.predictions;
    return [
      _header('Care summary to share with family'),
      pw.SizedBox(height: 16),
      _sectionTitle('Patient'),
      pw.Text(
        patient.name,
        style: pw.TextStyle(fontSize: 14, fontWeight: pw.FontWeight.bold),
      ),
      pw.SizedBox(height: 4),
      pw.Text('Room ${patient.roomNumber}'),
      if (patient.primaryDoctor != null)
        pw.Text('Attending: ${patient.primaryDoctor}'),
      pw.SizedBox(height: 16),
      _sectionTitle('Expected length of stay'),
      pw.Text(
        'Estimated hospital stay: ${p.hospitalLosDays.toStringAsFixed(1)} days',
        style: const pw.TextStyle(fontSize: 12),
      ),
      pw.SizedBox(height: 6),
      pw.Text(
        'This is an estimate and may change as your care team learns more.',
        style: const pw.TextStyle(fontSize: 9, color: PdfColors.grey600),
      ),
      pw.SizedBox(height: 16),
      _sectionTitle('Diagnoses'),
      ..._diagnosesBlock(patient),
      pw.SizedBox(height: 16),
      _sectionTitle('Medications'),
      ..._medicationsBlock(patient),
      pw.SizedBox(height: 16),
      _sectionTitle('Recommendations'),
      ..._recommendationsBlock(patient),
      pw.SizedBox(height: 20),
      _footer(dateStr),
    ];
  }

  static List<pw.Widget> _clinicalContent(
    PatientRecord patient,
    String dateStr,
  ) {
    // Clinician export uses the same family-safe layout when sharing externally.
    return _familyContent(patient, dateStr);
  }

  static pw.Widget _header(String subtitle) => pw.Column(
        crossAxisAlignment: pw.CrossAxisAlignment.start,
        children: [
          pw.Text(
            'Baptist Health Cardiogenic Shock Tracker',
            style: pw.TextStyle(
              fontSize: 18,
              fontWeight: pw.FontWeight.bold,
            ),
          ),
          pw.SizedBox(height: 4),
          pw.Text(
            subtitle,
            style: const pw.TextStyle(
              fontSize: 11,
              color: PdfColors.grey700,
            ),
          ),
        ],
      );

  static pw.Widget _sectionTitle(String text) => pw.Padding(
        padding: const pw.EdgeInsets.only(bottom: 6),
        child: pw.Text(
          text,
          style: pw.TextStyle(
            fontSize: 12,
            fontWeight: pw.FontWeight.bold,
            color: PdfColors.grey800,
          ),
        ),
      );

  static List<pw.Widget> _diagnosesBlock(PatientRecord patient) {
    if (patient.diagnoses.isEmpty) {
      return [
        pw.Text(
          'No diagnoses listed on file.',
          style: const pw.TextStyle(fontSize: 11, color: PdfColors.grey700),
        ),
      ];
    }
    return patient.diagnoses.map((d) {
      final label = d.classification == 'PRINCIPAL'
          ? 'Principal'
          : 'Secondary';
      return pw.Padding(
        padding: const pw.EdgeInsets.only(bottom: 6),
        child: pw.Text(
          '$label: ${d.text} (${d.code})',
          style: const pw.TextStyle(fontSize: 11),
        ),
      );
    }).toList();
  }

  static List<pw.Widget> _medicationsBlock(PatientRecord patient) {
    if (patient.medications.isEmpty) {
      return [
        pw.Text(
          'No active medications listed for this visit window.',
          style: const pw.TextStyle(fontSize: 11, color: PdfColors.grey700),
        ),
      ];
    }
    return patient.medications.map((m) {
      final parts = <String>[
        m.name,
        if (m.dosage.isNotEmpty) m.dosage,
        if (m.route.isNotEmpty) m.route,
      ];
      return pw.Padding(
        padding: const pw.EdgeInsets.only(bottom: 6),
        child: pw.Text(
          '• ${parts.join(' · ')}',
          style: const pw.TextStyle(fontSize: 11),
        ),
      );
    }).toList();
  }

  static List<pw.Widget> _recommendationsBlock(PatientRecord patient) {
    if (patient.recommendations.isEmpty) {
      return [
        pw.Text(
          'No recommendations at this time. Follow guidance from your care team.',
          style: const pw.TextStyle(fontSize: 11, color: PdfColors.grey700),
        ),
      ];
    }
    return patient.recommendations.map(
      (r) => pw.Padding(
        padding: const pw.EdgeInsets.only(bottom: 10),
        child: pw.Column(
          crossAxisAlignment: pw.CrossAxisAlignment.start,
          children: [
            pw.Text(
              r.title,
              style: pw.TextStyle(
                fontSize: 11,
                fontWeight: pw.FontWeight.bold,
              ),
            ),
            pw.SizedBox(height: 2),
            pw.Text(
              r.description,
              style: const pw.TextStyle(fontSize: 10),
            ),
            if (r.impact.isNotEmpty) ...[
              pw.SizedBox(height: 2),
              pw.Text(
                r.impact,
                style: const pw.TextStyle(
                  fontSize: 9,
                  color: PdfColors.grey600,
                ),
              ),
            ],
          ],
        ),
      ),
    ).toList();
  }

  static pw.Widget _footer(String dateStr) => pw.Text(
        'This summary is for communication support only and does not replace '
        'medical advice from your care team. It does not include clinical risk '
        'scores or predictions. Generated $dateStr.',
        style: const pw.TextStyle(fontSize: 9, color: PdfColors.grey600),
      );
}
