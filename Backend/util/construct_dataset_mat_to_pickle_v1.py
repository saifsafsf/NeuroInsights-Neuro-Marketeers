import scipy.io as io


def mat_to_dict(mat_filepath: str):
    dataset_dict = {}
    dataset_dict['subject_name'] = []

    matdata = io.loadmat(
        mat_filepath, 
        squeeze_me=True,
        struct_as_record=False
    )['sentenceData']

    for sent in matdata:
        word_data = sent.word
        if not isinstance(word_data, float):
            # sentence level:
            sent_obj = {'content': sent.content}
            sent_obj['sentence_level_EEG'] = {
                'mean_t1': sent.mean_t1,
                'mean_t2': sent.mean_t2,
                'mean_a1': sent.mean_a1,
                'mean_a2': sent.mean_a2,
                'mean_b1': sent.mean_b1,
                'mean_b2': sent.mean_b2,
                'mean_g1': sent.mean_g1,
                'mean_g2': sent.mean_g2
            }

            if mat_filepath.__contains__('SR'):
                sent_obj['answer_EEG'] = {
                    'answer_mean_t1': sent.answer_mean_t1,
                    'answer_mean_t2': sent.answer_mean_t2,
                    'answer_mean_a1': sent.answer_mean_a1,
                    'answer_mean_a2': sent.answer_mean_a2,
                    'answer_mean_b1': sent.answer_mean_b1,
                    'answer_mean_b2': sent.answer_mean_b2,
                    'answer_mean_g1': sent.answer_mean_g1,
                    'answer_mean_g2': sent.answer_mean_g2
                }

            # word level:
            sent_obj['word'] = []

            word_tokens_has_fixation = []
            word_tokens_with_mask = []
            word_tokens_all = []

            for word in word_data:
                word_obj = {'content': word.content}
                word_tokens_all.append(word.content)

                word_obj['nFixations'] = word.nFixations
                if word.nFixations > 0:
                    word_obj['word_level_EEG'] = {
                        'FFD': {
                            'FFD_t1': word.FFD_t1,
                            'FFD_t2': word.FFD_t2,
                            'FFD_a1': word.FFD_a1,
                            'FFD_a2': word.FFD_a2,
                            'FFD_b1': word.FFD_b1,
                            'FFD_b2': word.FFD_b2,
                            'FFD_g1': word.FFD_g1,
                            'FFD_g2': word.FFD_g2
                        }
                    }

                    word_obj['word_level_EEG']['TRT'] = {
                        'TRT_t1': word.TRT_t1,
                        'TRT_t2': word.TRT_t2,
                        'TRT_a1': word.TRT_a1,
                        'TRT_a2': word.TRT_a2,
                        'TRT_b1': word.TRT_b1,
                        'TRT_b2': word.TRT_b2,
                        'TRT_g1': word.TRT_g1,
                        'TRT_g2': word.TRT_g2
                    }

                    word_obj['word_level_EEG']['GD'] = {
                        'GD_t1': word.GD_t1,
                        'GD_t2': word.GD_t2,
                        'GD_a1': word.GD_a1,
                        'GD_a2': word.GD_a2,
                        'GD_b1': word.GD_b1,
                        'GD_b2': word.GD_b2,
                        'GD_g1': word.GD_g1,
                        'GD_g2': word.GD_g2
                    }

                    sent_obj['word'].append(word_obj)
                    word_tokens_has_fixation.append(word.content)
                    word_tokens_with_mask.append(word.content)

                else:
                    word_tokens_with_mask.append('[MASK]')
                    continue

            sent_obj['word_tokens_has_fixation'] = word_tokens_has_fixation
            sent_obj['word_tokens_with_mask'] = word_tokens_with_mask
            sent_obj['word_tokens_all'] = word_tokens_all

            dataset_dict['subject_name'].append(sent_obj)

        else:
            print(f'missing sent: content:{sent.content}, return None')
            dataset_dict['subject_name'].append(None)
            continue
        
    return dataset_dict