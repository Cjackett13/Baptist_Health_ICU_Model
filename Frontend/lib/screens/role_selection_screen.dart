import 'package:flutter/material.dart';

import '../main.dart';
import 'home_page.dart';
import 'patient_sign_in_screen.dart';

enum AppRole { clinician, patient }

class RoleSelectionScreen extends StatelessWidget {
  const RoleSelectionScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: BhColors.background,
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const SizedBox(height: 32),
              Image.asset(
                'assets/baptist_logo.png',
                height: 56,
                fit: BoxFit.contain,
              ),
              const SizedBox(height: 24),
              Text(
                'Baptist Health ICU',
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                      color: BhColors.ink,
                      fontWeight: FontWeight.w700,
                      fontFamily: 'Georgia',
                    ),
              ),
              const SizedBox(height: 8),
              Text(
                'Choose how you want to use this workspace',
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                      color: BhColors.slate,
                    ),
              ),
              const SizedBox(height: 40),
              _RoleCard(
                title: 'Nurse / Physician',
                subtitle:
                    'Full patient list, vitals from clinical data, and ML predictions '
                    '(SCAI, vasopressors, mortality, LOS, MCS, VA-ECMO).',
                icon: Icons.local_hospital_outlined,
                accent: BhColors.ink,
                onTap: () => _open(context, AppRole.clinician),
              ),
              const SizedBox(height: 16),
              _RoleCard(
                title: 'Patient',
                subtitle:
                    'Your predicted length of stay, personalized recommendations, '
                    'and prescribed medications with dosage details.',
                icon: Icons.person_outline,
                accent: BhColors.primary,
                onTap: () => _openPatientSignIn(context),
              ),
              const Spacer(),
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
    );
  }

  void _open(BuildContext context, AppRole role) {
    Navigator.of(context).pushReplacement(
      MaterialPageRoute<void>(
        builder: (_) => HomePage(role: role),
      ),
    );
  }

  void _openPatientSignIn(BuildContext context) {
    Navigator.of(context).pushReplacement(
      MaterialPageRoute<void>(
        builder: (_) => const PatientSignInScreen(),
      ),
    );
  }
}

class _RoleCard extends StatelessWidget {
  const _RoleCard({
    required this.title,
    required this.subtitle,
    required this.icon,
    required this.accent,
    required this.onTap,
  });

  final String title;
  final String subtitle;
  final IconData icon;
  final Color accent;
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
        child: Container(
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(20),
            border: Border.all(color: Colors.black.withValues(alpha: 0.08)),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withValues(alpha: 0.05),
                blurRadius: 16,
                offset: const Offset(0, 6),
              ),
            ],
          ),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: accent.withValues(alpha: 0.12),
                  borderRadius: BorderRadius.circular(14),
                ),
                child: Icon(icon, color: accent, size: 28),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      title,
                      style: const TextStyle(
                        fontSize: 17,
                        fontWeight: FontWeight.w700,
                        color: BhColors.ink,
                      ),
                    ),
                    const SizedBox(height: 6),
                    Text(
                      subtitle,
                      style: const TextStyle(
                        fontSize: 13,
                        color: Colors.black54,
                        height: 1.4,
                      ),
                    ),
                    const SizedBox(height: 12),
                    Row(
                      children: [
                        Text(
                          'Continue',
                          style: TextStyle(
                            fontSize: 13,
                            fontWeight: FontWeight.w700,
                            color: accent,
                          ),
                        ),
                        const SizedBox(width: 4),
                        Icon(Icons.arrow_forward_rounded,
                            size: 16, color: accent),
                      ],
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
