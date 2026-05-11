// lib/screens/home_care_screen.dart
//
// Prediction 3 — Home care recommendations for patient's family.
// Currently displays mock suggestions. When FastAPI is connected,
// replace _suggestions with the Claude API response from /home_care endpoint.

import 'package:flutter/material.dart';
import '../models/patient_prediction.dart';

class HomeCareScreen extends StatelessWidget {
  const HomeCareScreen({
    required this.patient,
    super.key,
  });

  final PatientRecord patient;

  @override
  Widget build(BuildContext context) {
    final suggestions = patient.predictions.homeCareSuggestions;

    return Scaffold(
      backgroundColor: const Color(0xFFF2F2F2),
      appBar: AppBar(
        backgroundColor: const Color(0xFF222831),
        foregroundColor: Colors.white,
        elevation: 0,
        title: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Home Care Plan',
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.w700,
                color: Colors.white,
              ),
            ),
            Text(
              patient.name,
              style: TextStyle(
                fontSize: 12,
                color: Colors.white.withOpacity(0.7),
              ),
            ),
          ],
        ),
        actions: [
          // Share button — nurse sends to family
          // Wire to email/PDF export later
          IconButton(
            onPressed: () => _showShareSheet(context),
            icon: const Icon(Icons.share_outlined),
            tooltip: 'Share with family',
          ),
        ],
      ),
      body: CustomScrollView(
        slivers: [
          // Intro banner
          SliverToBoxAdapter(
            child: _IntroBanner(patient: patient),
          ),

          // Suggestion cards
          SliverPadding(
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
            sliver: SliverList(
              delegate: SliverChildBuilderDelegate(
                (context, index) => Padding(
                  padding: const EdgeInsets.only(bottom: 12),
                  child: _SuggestionCard(
                    suggestion: suggestions[index],
                    index: index,
                  ),
                ),
                childCount: suggestions.length,
              ),
            ),
          ),

          // AI disclaimer
          const SliverToBoxAdapter(
            child: _AiDisclaimer(),
          ),

          const SliverToBoxAdapter(child: SizedBox(height: 32)),
        ],
      ),
    );
  }

  void _showShareSheet(BuildContext context) {
    showModalBottomSheet<void>(
      context: context,
      backgroundColor: Colors.white,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (context) => Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Share home care plan',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.w700,
              ),
            ),
            const SizedBox(height: 6),
            Text(
              'Send personalized instructions to ${patient.name}\'s family',
              style: const TextStyle(
                fontSize: 13,
                color: Colors.black54,
              ),
            ),
            const SizedBox(height: 20),
            _ShareOption(
              icon: Icons.picture_as_pdf_outlined,
              label: 'Export as PDF',
              subtitle: 'Print-ready format for discharge packet',
              onTap: () {
                Navigator.pop(context);
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(
                    content: Text(
                        'PDF export — connect FastAPI /export endpoint'),
                  ),
                );
              },
            ),
            const SizedBox(height: 12),
            _ShareOption(
              icon: Icons.email_outlined,
              label: 'Send via email',
              subtitle: 'Delivered directly to family\'s inbox',
              onTap: () {
                Navigator.pop(context);
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(
                    content:
                        Text('Email — connect FastAPI /email endpoint'),
                  ),
                );
              },
            ),
            const SizedBox(height: 24),
          ],
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// INTRO BANNER
// ─────────────────────────────────────────────────────────────────────────────
class _IntroBanner extends StatelessWidget {
  const _IntroBanner({required this.patient});
  final PatientRecord patient;

  @override
  Widget build(BuildContext context) {
    final readPct =
        (patient.predictions.readmissionRisk * 100).round();

    return Container(
      margin: const EdgeInsets.all(16),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF222831),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: const Color(0xFF4A9E6A).withOpacity(0.2),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: const Icon(
                  Icons.home_outlined,
                  color: Color(0xFF4A9E6A),
                  size: 20,
                ),
              ),
              const SizedBox(width: 10),
              const Expanded(
                child: Text(
                  'Personalized care plan',
                  style: TextStyle(
                    fontSize: 15,
                    fontWeight: FontWeight.w700,
                    color: Colors.white,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          Text(
            'Based on ${patient.name}\'s clinical profile, our AI has identified '
            'the following recommendations to help their family support recovery at home '
            'and reduce a $readPct% readmission risk.',
            style: TextStyle(
              fontSize: 13,
              color: Colors.white.withOpacity(0.75),
              height: 1.5,
            ),
          ),
          const SizedBox(height: 12),
          Container(
            padding:
                const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
            decoration: BoxDecoration(
              color: const Color(0xFF4A9E6A).withOpacity(0.15),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Icon(Icons.auto_awesome,
                    color: Color(0xFF4A9E6A), size: 13),
                const SizedBox(width: 5),
                Text(
                  'Generated by Baptist Health AI · ${_formatDate(patient.predictions.lastUpdated)}',
                  style: const TextStyle(
                    fontSize: 10,
                    color: Color(0xFF4A9E6A),
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  String _formatDate(DateTime dt) =>
      '${dt.month}/${dt.day}/${dt.year}';
}

// ─────────────────────────────────────────────────────────────────────────────
// SUGGESTION CARD
// One recommendation — category icon, title, description, impact badge.
// ─────────────────────────────────────────────────────────────────────────────
class _SuggestionCard extends StatefulWidget {
  const _SuggestionCard({
    required this.suggestion,
    required this.index,
  });

  final HomeCareSuggestion suggestion;
  final int index;

  @override
  State<_SuggestionCard> createState() => _SuggestionCardState();
}

class _SuggestionCardState extends State<_SuggestionCard> {
  bool _expanded = false;

  @override
  Widget build(BuildContext context) {
    final s = widget.suggestion;
    final icon = categoryIcon(s.category);

    return GestureDetector(
      onTap: () => setState(() => _expanded = !_expanded),
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: Colors.black.withOpacity(0.07)),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.03),
              blurRadius: 8,
              offset: const Offset(0, 2),
            ),
          ],
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                // Category emoji icon
                Container(
                  width: 40,
                  height: 40,
                  decoration: BoxDecoration(
                    color: const Color(0xFFF2F2F2),
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Center(
                    child: Text(icon, style: const TextStyle(fontSize: 20)),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        s.title,
                        style: const TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.w700,
                          color: Colors.black87,
                        ),
                      ),
                      const SizedBox(height: 2),
                      Text(
                        _categoryLabel(s.category),
                        style: const TextStyle(
                          fontSize: 11,
                          color: Colors.black45,
                        ),
                      ),
                    ],
                  ),
                ),
                Icon(
                  _expanded
                      ? Icons.keyboard_arrow_up_rounded
                      : Icons.keyboard_arrow_down_rounded,
                  color: Colors.black38,
                ),
              ],
            ),

            // Impact badge — always visible (single line; scales down if tight)
            const SizedBox(height: 10),
            LayoutBuilder(
              builder: (context, constraints) {
                return Container(
                  constraints:
                      BoxConstraints(maxWidth: constraints.maxWidth),
                  padding: const EdgeInsets.symmetric(
                      horizontal: 10, vertical: 5),
                  decoration: BoxDecoration(
                    color: const Color(0xFF4A9E6A).withOpacity(0.10),
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const Icon(Icons.trending_down,
                          color: Color(0xFF4A9E6A), size: 13),
                      const SizedBox(width: 4),
                      Flexible(
                        child: FittedBox(
                          fit: BoxFit.scaleDown,
                          alignment: Alignment.centerLeft,
                          child: Text(
                            s.impact,
                            maxLines: 1,
                            softWrap: false,
                            style: const TextStyle(
                              fontSize: 11,
                              color: Color(0xFF4A9E6A),
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                );
              },
            ),

            // Expanded description
            AnimatedCrossFade(
              firstChild: const SizedBox(width: double.infinity),
              secondChild: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const SizedBox(height: 12),
                  const Divider(height: 1, color: Color(0xFFF0F0F0)),
                  const SizedBox(height: 12),
                  Text(
                    s.description,
                    style: const TextStyle(
                      fontSize: 13,
                      color: Colors.black87,
                      height: 1.6,
                    ),
                  ),
                ],
              ),
              crossFadeState: _expanded
                  ? CrossFadeState.showSecond
                  : CrossFadeState.showFirst,
              duration: const Duration(milliseconds: 200),
            ),
          ],
        ),
      ),
    );
  }

  String _categoryLabel(String category) {
    switch (category) {
      case 'diet':
        return 'Nutrition';
      case 'exercise':
        return 'Physical activity';
      case 'medication':
        return 'Medication management';
      case 'monitoring':
        return 'Health monitoring';
      case 'lifestyle':
        return 'Lifestyle';
      default:
        return 'General';
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// SHARE OPTION ROW
// ─────────────────────────────────────────────────────────────────────────────
class _ShareOption extends StatelessWidget {
  const _ShareOption({
    required this.icon,
    required this.label,
    required this.subtitle,
    required this.onTap,
  });

  final IconData icon;
  final String label;
  final String subtitle;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: const Color(0xFFF5F5F5),
          borderRadius: BorderRadius.circular(12),
        ),
        child: Row(
          children: [
            Icon(icon, color: const Color(0xFF222831), size: 22),
            const SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    label,
                    style: const TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.w600,
                      color: Colors.black87,
                    ),
                  ),
                  Text(
                    subtitle,
                    style: const TextStyle(
                      fontSize: 12,
                      color: Colors.black45,
                    ),
                  ),
                ],
              ),
            ),
            const Icon(Icons.chevron_right_rounded,
                color: Colors.black26),
          ],
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// AI DISCLAIMER
// Required for any clinical AI tool — makes clear this is decision support
// not a replacement for clinical judgment.
// ─────────────────────────────────────────────────────────────────────────────
class _AiDisclaimer extends StatelessWidget {
  const _AiDisclaimer();

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.fromLTRB(16, 0, 16, 16),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.04),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.black12),
      ),
      child: const Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(Icons.info_outline, size: 16, color: Colors.black38),
          SizedBox(width: 8),
          Expanded(
            child: Text(
              'These recommendations are generated by an AI model and are intended '
              'as decision support only. They do not replace the clinical judgment of '
              'a licensed healthcare professional. Always consult your care team before '
              'making changes to medications or treatment plans.',
              style: TextStyle(
                fontSize: 11,
                color: Colors.black45,
                height: 1.5,
              ),
            ),
          ),
        ],
      ),
    );
  }
}