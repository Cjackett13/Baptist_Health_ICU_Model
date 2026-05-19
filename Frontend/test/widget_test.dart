import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:flutter_application_1/main.dart';
import 'package:flutter_application_1/screens/role_selection_screen.dart';

void main() {
  testWidgets('App loads role selection screen', (WidgetTester tester) async {
    await tester.pumpWidget(const MyApp());

    expect(find.byType(MaterialApp), findsOneWidget);
    expect(find.byType(RoleSelectionScreen), findsOneWidget);
    expect(find.text('Nurse / Physician'), findsOneWidget);
    expect(find.text('Patient'), findsOneWidget);
  });
}
