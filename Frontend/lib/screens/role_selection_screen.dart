import 'package:flutter/material.dart';

import '../theme/app_colors.dart';
import '../widgets/bh_branded_header.dart';
import 'home_page.dart';
import 'patient_hospital_screen.dart';

enum AppRole { clinician, patient }

class RoleSelectionScreen extends StatelessWidget {
  const RoleSelectionScreen({super.key});

  static const double _boxGap = 32;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: BhColors.background,
      body: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          const BhBrandedHeader(
            subtitle: 'Choose how you want to use this workspace',
          ),
          Expanded(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 24),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  const SizedBox(height: 24),
                  Expanded(
                    child: LayoutBuilder(
                      builder: (context, constraints) {
                        final maxW = constraints.maxWidth;
                        final maxH = constraints.maxHeight;
                        final boxSize = _squareSize(maxW, maxH);

                        return Center(
                          child: Row(
                            mainAxisSize: MainAxisSize.min,
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              SizedBox(
                                width: boxSize,
                                height: boxSize,
                                child: _RoleCard(
                                  title: 'Nurse / Physician',
                                  subtitle:
                                      'Patient list, vitals, and ML predictions.',
                                  icon: Icons.local_hospital_outlined,
                                  iconColor: const Color(0xFFE05A5A),
                                  onTap: () => _openClinician(context),
                                ),
                              ),
                              const SizedBox(width: _boxGap),
                              SizedBox(
                                width: boxSize,
                                height: boxSize,
                                child: _RoleCard(
                                  title: 'Patient',
                                  subtitle:
                                      'Your care page, meds, and care assistant chat.',
                                  icon: Icons.person_outline,
                                  iconColor: const Color(0xFF9BD67D),
                                  onTap: () => _openPatientSignIn(context),
                                ),
                              ),
                            ],
                          ),
                        );
                      },
                    ),
                  ),
                  Text(
                    'Demo data sourced from project parquet tables (person, encounter, '
                    'clinical_event, diagnosis, medication_admin, scai_stage_hourly).',
                    textAlign: TextAlign.center,
                    style: Theme.of(context).textTheme.bodySmall?.copyWith(
                          color: Colors.black38,
                          fontSize: 11,
                        ),
                  ),
                  const SizedBox(height: 24),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  /// Equal square tiles, centered, with fixed gap between them.
  double _squareSize(double maxWidth, double maxHeight) {
    const maxBox = 286.0; // 220 × 1.3
    const minBox = 182.0; // 140 × 1.3
    final fromWidth = (maxWidth - _boxGap) / 2;
    final fromHeight = maxHeight;
    return (fromWidth < fromHeight ? fromWidth : fromHeight)
        .clamp(minBox, maxBox);
  }

  void _openClinician(BuildContext context) {
    Navigator.of(context).pushReplacement(
      MaterialPageRoute<void>(
        builder: (_) => const HomePage(),
      ),
    );
  }

  void _openPatientSignIn(BuildContext context) {
    Navigator.of(context).pushReplacement(
      MaterialPageRoute<void>(
        builder: (_) => const PatientHospitalScreen(),
      ),
    );
  }
}

class _RoleCard extends StatelessWidget {
  const _RoleCard({
    required this.title,
    required this.subtitle,
    required this.icon,
    required this.iconColor,
    required this.onTap,
  });

  final String title;
  final String subtitle;
  final IconData icon;
  final Color iconColor;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return Material(
      color: Colors.white,
      borderRadius: BorderRadius.circular(20),
      elevation: 0,
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(20),
        splashColor: BhColors.ink.withValues(alpha: 0.08),
        highlightColor: BhColors.ink.withValues(alpha: 0.04),
        child: Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(20),
            border: Border.all(color: BhColors.ink.withValues(alpha: 0.12)),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withValues(alpha: 0.06),
                blurRadius: 16,
                offset: const Offset(0, 6),
              ),
            ],
          ),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: iconColor.withValues(alpha: 0.14),
                  borderRadius: BorderRadius.circular(14),
                ),
                child: Icon(icon, color: iconColor, size: 36),
              ),
              const SizedBox(height: 14),
              Text(
                title,
                textAlign: TextAlign.center,
                style: const TextStyle(
                  fontSize: 15,
                  fontWeight: FontWeight.w700,
                  color: BhColors.ink,
                ),
              ),
              const SizedBox(height: 6),
              Text(
                subtitle,
                textAlign: TextAlign.center,
                maxLines: 3,
                overflow: TextOverflow.ellipsis,
                style: TextStyle(
                  fontSize: 11,
                  color: BhColors.ink.withValues(alpha: 0.65),
                  height: 1.35,
                ),
              ),
              const SizedBox(height: 12),
              const Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    'Continue',
                    style: TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.w700,
                      color: BhColors.ink,
                    ),
                  ),
                  SizedBox(width: 4),
                  Icon(
                    Icons.arrow_forward_rounded,
                    size: 14,
                    color: BhColors.ink,
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}
