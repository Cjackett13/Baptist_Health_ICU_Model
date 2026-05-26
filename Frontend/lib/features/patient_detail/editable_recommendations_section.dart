import 'package:flutter/material.dart';

import '../../models/patient_prediction.dart';
import '../../services/patient_repository.dart';
import '../../theme/app_colors.dart';

/// Clinician editor for patient/family recommendations (syncs to patient POV).
class EditableRecommendationsSection extends StatefulWidget {
  const EditableRecommendationsSection({
    required this.patientId,
    required this.initialRecommendations,
    super.key,
  });

  final String patientId;
  final List<HomeCareSuggestion> initialRecommendations;

  @override
  State<EditableRecommendationsSection> createState() =>
      _EditableRecommendationsSectionState();
}

class _EditableRecommendationsSectionState
    extends State<EditableRecommendationsSection> {
  late List<HomeCareSuggestion> _items;

  @override
  void initState() {
    super.initState();
    _items = List<HomeCareSuggestion>.from(widget.initialRecommendations);
  }

  @override
  void didUpdateWidget(EditableRecommendationsSection oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.initialRecommendations != widget.initialRecommendations) {
      _items = List<HomeCareSuggestion>.from(widget.initialRecommendations);
    }
  }

  void _persist() {
    PatientRepository.instance.updateRecommendations(widget.patientId, _items);
  }

  Future<void> _addOrEdit({HomeCareSuggestion? existing, int? index}) async {
    final result = await showDialog<HomeCareSuggestion>(
      context: context,
      builder: (ctx) => _RecommendationEditorDialog(existing: existing),
    );
    if (result == null || !mounted) return;
    setState(() {
      if (index != null) {
        _items[index] = result;
      } else {
        _items.add(result);
      }
    });
    _persist();
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            index != null ? 'Recommendation updated' : 'Recommendation added',
          ),
          behavior: SnackBarBehavior.floating,
        ),
      );
    }
  }

  void _delete(int index) {
    setState(() => _items.removeAt(index));
    _persist();
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('Recommendation removed'),
        behavior: SnackBarBehavior.floating,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Text(
          'Edits are saved for this session and appear on the patient\'s care page.',
          style: TextStyle(
            fontSize: 11,
            color: Colors.black.withValues(alpha: 0.5),
            height: 1.35,
          ),
        ),
        const SizedBox(height: 12),
        if (_items.isEmpty)
          const Padding(
            padding: EdgeInsets.symmetric(vertical: 8),
            child: Text(
              'No recommendations yet. Add guidance for the patient and family.',
              style: TextStyle(fontSize: 13, color: Colors.black45),
            ),
          )
        else
          ...List.generate(_items.length, (index) {
            final s = _items[index];
            return Padding(
              padding: const EdgeInsets.only(bottom: 10),
              child: Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(
                    color: Colors.black.withValues(alpha: 0.08),
                  ),
                ),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      categoryIcon(s.category),
                      style: const TextStyle(fontSize: 22),
                    ),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            s.title,
                            style: const TextStyle(
                              fontSize: 14,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                          const SizedBox(height: 4),
                          Text(
                            s.description,
                            style: const TextStyle(
                              fontSize: 12,
                              color: Colors.black54,
                              height: 1.35,
                            ),
                          ),
                          if (s.impact.isNotEmpty) ...[
                            const SizedBox(height: 4),
                            Text(
                              s.impact,
                              style: const TextStyle(
                                fontSize: 11,
                                fontWeight: FontWeight.w600,
                                color: Color(0xFF4A9E6A),
                              ),
                            ),
                          ],
                        ],
                      ),
                    ),
                    IconButton(
                      tooltip: 'Edit',
                      onPressed: () => _addOrEdit(
                        existing: s,
                        index: index,
                      ),
                      icon: const Icon(Icons.edit_outlined, size: 20),
                    ),
                    IconButton(
                      tooltip: 'Delete',
                      onPressed: () => _delete(index),
                      icon: const Icon(
                        Icons.delete_outline,
                        size: 20,
                        color: Color(0xFFE05A5A),
                      ),
                    ),
                  ],
                ),
              ),
            );
          }),
        const SizedBox(height: 8),
        OutlinedButton.icon(
          onPressed: () => _addOrEdit(),
          icon: const Icon(Icons.add, size: 18),
          label: const Text('Add recommendation'),
          style: OutlinedButton.styleFrom(
            foregroundColor: BhColors.primary,
            side: BorderSide(color: BhColors.primary.withValues(alpha: 0.5)),
          ),
        ),
      ],
    );
  }
}

class _RecommendationEditorDialog extends StatefulWidget {
  const _RecommendationEditorDialog({this.existing});
  final HomeCareSuggestion? existing;

  @override
  State<_RecommendationEditorDialog> createState() =>
      _RecommendationEditorDialogState();
}

class _RecommendationEditorDialogState
    extends State<_RecommendationEditorDialog> {
  static const _categories = [
    'monitoring',
    'medication',
    'lifestyle',
    'diet',
    'exercise',
  ];

  late final TextEditingController _titleController;
  late final TextEditingController _descriptionController;
  late final TextEditingController _impactController;
  late String _category;

  @override
  void initState() {
    super.initState();
    final e = widget.existing;
    _titleController = TextEditingController(text: e?.title ?? '');
    _descriptionController = TextEditingController(text: e?.description ?? '');
    _impactController = TextEditingController(text: e?.impact ?? '');
    _category = e?.category ?? 'monitoring';
  }

  @override
  void dispose() {
    _titleController.dispose();
    _descriptionController.dispose();
    _impactController.dispose();
    super.dispose();
  }

  void _save() {
    final title = _titleController.text.trim();
    final description = _descriptionController.text.trim();
    if (title.isEmpty || description.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Title and description are required')),
      );
      return;
    }
    Navigator.pop(
      context,
      HomeCareSuggestion(
        category: _category,
        title: title,
        description: description,
        impact: _impactController.text.trim(),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final isEdit = widget.existing != null;
    return AlertDialog(
      title: Text(isEdit ? 'Edit recommendation' : 'Add recommendation'),
      content: SingleChildScrollView(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            DropdownButtonFormField<String>(
              value: _category,
              decoration: const InputDecoration(labelText: 'Category'),
              items: _categories
                  .map(
                    (c) => DropdownMenuItem(
                      value: c,
                      child: Text(
                        '${categoryIcon(c)} ${_categoryLabel(c)}',
                      ),
                    ),
                  )
                  .toList(),
              onChanged: (v) {
                if (v != null) setState(() => _category = v);
              },
            ),
            const SizedBox(height: 12),
            TextField(
              controller: _titleController,
              decoration: const InputDecoration(labelText: 'Title'),
              textCapitalization: TextCapitalization.sentences,
            ),
            const SizedBox(height: 12),
            TextField(
              controller: _descriptionController,
              decoration: const InputDecoration(labelText: 'Description'),
              minLines: 2,
              maxLines: 4,
              textCapitalization: TextCapitalization.sentences,
            ),
            const SizedBox(height: 12),
            TextField(
              controller: _impactController,
              decoration: const InputDecoration(
                labelText: 'Expected benefit (optional)',
              ),
              textCapitalization: TextCapitalization.sentences,
            ),
          ],
        ),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context),
          child: const Text('Cancel'),
        ),
        FilledButton(
          onPressed: _save,
          style: FilledButton.styleFrom(backgroundColor: BhColors.primary),
          child: Text(isEdit ? 'Save' : 'Add'),
        ),
      ],
    );
  }

  static String _categoryLabel(String c) {
    if (c.isEmpty) return c;
    return '${c[0].toUpperCase()}${c.substring(1)}';
  }
}
