import torch


def ctc_alignment(log_probs, targets, input_lengths, target_lengths, blank=0, ignore_id=-1):
    targets[targets==ignore_id] = 0

    batch_size, input_time_size, _ = log_probs.size()
    B_arange = torch.arange(batch_size, device=input_lengths.device)

    _t_a_r_g_e_t_s_ = torch.cat(
        (
            torch.stack([torch.full_like(targets, blank), targets], dim=-1).flatten(start_dim=1),
            torch.full_like(targets[:, :1], blank),
        ),
        dim=-1,
    )
    diff_labels = torch.cat(
        (
            torch.as_tensor([[False, False]], device=targets.device).expand(batch_size, -1),
            _t_a_r_g_e_t_s_[:, 2:] != _t_a_r_g_e_t_s_[:, :-2],
        ),
        dim=1,
    )

    zero_padding, zero = 2, torch.tensor(float("-inf"), device=log_probs.device, dtype=log_probs.dtype)
    padded_t = zero_padding + _t_a_r_g_e_t_s_.size(-1)
    best_score = torch.full((batch_size, padded_t), zero, device=log_probs.device, dtype=log_probs.dtype)
    best_score[:, zero_padding + 0] = log_probs[:, 0, blank]
    best_score[:, zero_padding + 1] = log_probs[B_arange, 0, _t_a_r_g_e_t_s_[:, 1]]

    backpointers_shape = [batch_size, input_time_size, padded_t]
    backpointers = torch.zeros(backpointers_shape, device=log_probs.device, dtype=targets.dtype)

    for t in range(1, input_time_size):
        prev = torch.stack([best_score[:, 2:], best_score[:, 1:-1], torch.where(diff_labels, best_score[:, :-2], zero)])
        prev_max_value, prev_max_idx = prev.max(dim=0)
        best_score[:, zero_padding:] = log_probs[:, t].gather(-1, _t_a_r_g_e_t_s_) + prev_max_value
        backpointers[:, t, zero_padding:] = prev_max_idx

    l1l2 = best_score.gather(
        -1, torch.stack([zero_padding + target_lengths * 2 - 1, zero_padding + target_lengths * 2], dim=-1)
    )

    path = torch.zeros((batch_size, input_time_size), device=best_score.device, dtype=torch.long)
    path[B_arange, input_lengths - 1] = zero_padding + target_lengths * 2 - 1 + l1l2.argmax(dim=-1)

    for t in range(input_time_size - 1, 0, -1):
        indices = path[:, t]
        prev_max_idx = backpointers[B_arange, t, indices]
        path[:, t - 1] += indices - prev_max_idx

    alignments = _t_a_r_g_e_t_s_.gather(dim=-1, index=(path - zero_padding).clamp(min=0))
    return alignments
