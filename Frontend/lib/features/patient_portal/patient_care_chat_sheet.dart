import 'package:flutter/material.dart';

import '../../models/patient_prediction.dart';
import '../../services/patient_care_assistant.dart';
import '../../theme/app_colors.dart';

class ChatMessage {
  const ChatMessage({
    required this.text,
    required this.isUser,
    required this.time,
  });

  final String text;
  final bool isUser;
  final DateTime time;
}

/// Bottom sheet chat for patient questions about meds, costs, and recovery.
class PatientCareChatSheet extends StatefulWidget {
  const PatientCareChatSheet({required this.patient, super.key});

  final PatientRecord patient;

  static Future<void> show(BuildContext context, PatientRecord patient) {
    return showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (ctx) => PatientCareChatSheet(patient: patient),
    );
  }

  @override
  State<PatientCareChatSheet> createState() => _PatientCareChatSheetState();
}

class _PatientCareChatSheetState extends State<PatientCareChatSheet> {
  final List<ChatMessage> _messages = [];
  final TextEditingController _controller = TextEditingController();
  final ScrollController _scroll = ScrollController();

  @override
  void initState() {
    super.initState();
    _messages.add(
      ChatMessage(
        text: 'Hi ${widget.patient.name.split(' ').first}! I can answer questions about '
            'your medications, typical costs, and recovery tips on your chart. '
            'I am not a doctor — call your nurse for urgent concerns.',
        isUser: false,
        time: DateTime.now(),
      ),
    );
  }

  @override
  void dispose() {
    _controller.dispose();
    _scroll.dispose();
    super.dispose();
  }

  void _send([String? preset]) {
    final text = (preset ?? _controller.text).trim();
    if (text.isEmpty) return;
    _controller.clear();
    setState(() {
      _messages.add(ChatMessage(text: text, isUser: true, time: DateTime.now()));
      _messages.add(
        ChatMessage(
          text: PatientCareAssistant.reply(widget.patient, text),
          isUser: false,
          time: DateTime.now(),
        ),
      );
    });
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scroll.hasClients) {
        _scroll.animateTo(
          _scroll.position.maxScrollExtent,
          duration: const Duration(milliseconds: 250),
          curve: Curves.easeOut,
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final height = MediaQuery.sizeOf(context).height * 0.78;
    return Container(
      height: height,
      decoration: const BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      child: Column(
        children: [
          const SizedBox(height: 10),
          Container(
            width: 40,
            height: 4,
            decoration: BoxDecoration(
              color: Colors.black12,
              borderRadius: BorderRadius.circular(2),
            ),
          ),
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 14, 8, 8),
            child: Row(
              children: [
                const Icon(Icons.chat_bubble_outline, color: BhColors.primary),
                const SizedBox(width: 10),
                const Expanded(
                  child: Text(
                    'Care assistant',
                    style: TextStyle(
                      fontSize: 17,
                      fontWeight: FontWeight.w700,
                      color: BhColors.ink,
                    ),
                  ),
                ),
                IconButton(
                  onPressed: () => Navigator.pop(context),
                  icon: const Icon(Icons.close),
                ),
              ],
            ),
          ),
          const Divider(height: 1),
          Padding(
            padding: const EdgeInsets.fromLTRB(12, 10, 12, 0),
            child: Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                _QuickChip(
                  label: 'My medications',
                  onTap: () => _send('What medications am I taking?'),
                ),
                _QuickChip(
                  label: 'Medication costs',
                  onTap: () => _send('What do my medications cost?'),
                ),
                _QuickChip(
                  label: 'Health tips',
                  onTap: () => _send('Recommendations to improve my health'),
                ),
              ],
            ),
          ),
          Expanded(
            child: ListView.builder(
              controller: _scroll,
              padding: const EdgeInsets.all(16),
              itemCount: _messages.length,
              itemBuilder: (context, index) {
                final m = _messages[index];
                return _Bubble(message: m);
              },
            ),
          ),
          const Divider(height: 1),
          SafeArea(
            top: false,
            child: Padding(
              padding: const EdgeInsets.fromLTRB(12, 10, 12, 12),
              child: Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: _controller,
                      minLines: 1,
                      maxLines: 3,
                      decoration: InputDecoration(
                        hintText: 'Ask about meds, costs, or recovery…',
                        filled: true,
                        fillColor: const Color(0xFFF2F2F2),
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(24),
                          borderSide: BorderSide.none,
                        ),
                        contentPadding: const EdgeInsets.symmetric(
                          horizontal: 16,
                          vertical: 12,
                        ),
                      ),
                      onSubmitted: (_) => _send(),
                    ),
                  ),
                  const SizedBox(width: 8),
                  FilledButton(
                    onPressed: _send,
                    style: FilledButton.styleFrom(
                      backgroundColor: BhColors.primary,
                      foregroundColor: Colors.white,
                      padding: const EdgeInsets.all(14),
                      shape: const CircleBorder(),
                    ),
                    child: const Icon(Icons.send_rounded, size: 20),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _QuickChip extends StatelessWidget {
  const _QuickChip({required this.label, required this.onTap});
  final String label;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return ActionChip(
      label: Text(label, style: const TextStyle(fontSize: 12)),
      onPressed: onTap,
      backgroundColor: BhColors.primary.withValues(alpha: 0.1),
      side: BorderSide(color: BhColors.primary.withValues(alpha: 0.25)),
    );
  }
}

class _Bubble extends StatelessWidget {
  const _Bubble({required this.message});
  final ChatMessage message;

  @override
  Widget build(BuildContext context) {
    final isUser = message.isUser;
    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.only(bottom: 10),
        constraints: BoxConstraints(
          maxWidth: MediaQuery.sizeOf(context).width * 0.82,
        ),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
        decoration: BoxDecoration(
          color: isUser ? BhColors.primary : const Color(0xFFF2F2F2),
          borderRadius: BorderRadius.only(
            topLeft: const Radius.circular(14),
            topRight: const Radius.circular(14),
            bottomLeft: Radius.circular(isUser ? 14 : 4),
            bottomRight: Radius.circular(isUser ? 4 : 14),
          ),
        ),
        child: Text(
          message.text,
          style: TextStyle(
            fontSize: 13,
            height: 1.4,
            color: isUser ? Colors.white : Colors.black87,
          ),
        ),
      ),
    );
  }
}
