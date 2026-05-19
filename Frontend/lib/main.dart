// lib/main.dart
import 'package:flutter/material.dart';

import 'screens/role_selection_screen.dart';

void main() {
  runApp(const MyApp());
}

abstract final class BhColors {
  static const primary = Color(0xFF7BC74D);
  static const ink = Color(0xFF222831);
  static const slate = Color(0xFF393E46);
  static const background = Color(0xFFEEEEEE);
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Baptist Health ICU',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        scaffoldBackgroundColor: BhColors.background,
        colorScheme: const ColorScheme.light(
          primary: BhColors.primary,
          onPrimary: Colors.white,
          surface: Colors.white,
          onSurface: BhColors.ink,
          secondary: BhColors.slate,
          onSecondary: Colors.white,
        ),
      ),
      home: const RoleSelectionScreen(),
    );
  }
}
