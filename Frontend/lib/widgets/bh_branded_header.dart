import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../theme/app_colors.dart';

/// Navy top bar with Baptist logo — used on role select, clinician home, etc.
class BhBrandedHeader extends StatelessWidget implements PreferredSizeWidget {
  const BhBrandedHeader({
    this.subtitle,
    this.trailing,
    super.key,
  });

  final String? subtitle;
  final Widget? trailing;

  static const double _barHeight = 72;

  @override
  Size get preferredSize => const Size.fromHeight(_barHeight);

  @override
  Widget build(BuildContext context) {
    final top = MediaQuery.paddingOf(context).top;

    return AnnotatedRegion<SystemUiOverlayStyle>(
      value: SystemUiOverlayStyle.light.copyWith(
        statusBarColor: BhColors.ink,
      ),
      child: Container(
        width: double.infinity,
        padding: EdgeInsets.fromLTRB(16, top + 12, 16, 16),
        decoration: const BoxDecoration(
          color: BhColors.ink,
          boxShadow: [
            BoxShadow(
              color: Color(0x33000000),
              blurRadius: 12,
              offset: Offset(0, 4),
            ),
          ],
        ),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            Image.asset(
              'assets/baptist_logo.png',
              height: 44,
              fit: BoxFit.contain,
              filterQuality: FilterQuality.high,
            ),
            const SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Text(
                    'Baptist Health Cardiogenic Shock Tracker',
                    style: Theme.of(context).textTheme.titleLarge?.copyWith(
                          color: Colors.white,
                          fontWeight: FontWeight.w700,
                          fontFamily: 'Georgia',
                          letterSpacing: 0.2,
                          fontSize: 16,
                        ),
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                  ),
                  if (subtitle != null && subtitle!.isNotEmpty) ...[
                    const SizedBox(height: 2),
                    Text(
                      subtitle!,
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                            color: Colors.white.withValues(alpha: 0.78),
                            fontWeight: FontWeight.w500,
                          ),
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                    ),
                  ],
                ],
              ),
            ),
            if (trailing != null) trailing!,
          ],
        ),
      ),
    );
  }
}

/// Scaffold wrapper: navy branded header + body below.
class BhBrandedScaffold extends StatelessWidget {
  const BhBrandedScaffold({
    required this.body,
    this.subtitle,
    this.trailing,
    this.bottomNavigationBar,
    super.key,
  });

  final Widget body;
  final String? subtitle;
  final Widget? trailing;
  final Widget? bottomNavigationBar;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: BhColors.background,
      body: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          BhBrandedHeader(subtitle: subtitle, trailing: trailing),
          Expanded(child: body),
        ],
      ),
      bottomNavigationBar: bottomNavigationBar,
    );
  }
}
