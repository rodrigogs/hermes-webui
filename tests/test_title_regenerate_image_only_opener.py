"""Title regeneration on a conversation whose opening turn carries no text.

Observed: clicking "regenerate title" answered `Could not generate a better title
(empty_user_message)` on a session that plainly had user messages. The cause is that a user turn
which is only an image (a pasted screenshot with no caption) yields '' from `_message_text` -- which
is correct, there is no text in it -- and both the first-exchange scanner and the caller treated "no
text" as "no message". The assistant's reply to that screenshot is real substance and is exactly what
the title should come from.
"""

from unittest.mock import MagicMock, patch

# Imported for its side effect: `patch('api.profiles....')` resolves attributes by walking the module
# path, and `api.profiles` is only bound on the `api` package once it has been imported. The function
# under test imports it lazily, inside the call.
import api.profiles  # noqa: F401
from api.streaming import (
    _first_exchange_snippets,
    _latest_exchange_snippets,
    _message_has_nontext_content,
    _message_text,
    generate_session_title_for_session,
)


def _image_part(url='data:image/png;base64,AAA'):
    return {'type': 'image_url', 'image_url': {'url': url}}


# --------------------------------------------------------------------- _message_has_nontext_content

def test_plain_string_has_no_nontext_content():
    assert _message_has_nontext_content('hello') is False


def test_text_only_blocks_have_no_nontext_content():
    assert _message_has_nontext_content([{'type': 'text', 'text': 'hello'}]) is False


def test_image_block_is_nontext_content():
    assert _message_has_nontext_content([_image_part()]) is True


def test_untyped_block_is_not_treated_as_media():
    # An untyped part is text by `_message_text`'s own rule; it must not be counted as media here or
    # a plain-text message would take the media path.
    assert _message_has_nontext_content([{'text': 'hello'}]) is False


# --------------------------------------------------------------------- the scanners

def test_image_only_opener_still_yields_the_assistant_answer():
    msgs = [
        {'role': 'user', 'content': [_image_part()]},
        {'role': 'assistant', 'content': 'This is a screenshot of a failing build.'},
    ]
    # No user text -- there is none in the message -- but the exchange is not empty.
    assert _message_text(msgs[0]['content']) == ''
    user_text, asst_text = _first_exchange_snippets(msgs)
    assert user_text == ''
    assert asst_text == 'This is a screenshot of a failing build.'


def test_first_exchange_stops_at_the_second_user_message():
    msgs = [
        {'role': 'user', 'content': [_image_part()]},
        {'role': 'assistant', 'content': 'first answer'},
        {'role': 'user', 'content': 'second question'},
        {'role': 'assistant', 'content': 'second answer'},
    ]
    _, asst_text = _first_exchange_snippets(msgs)
    assert asst_text == 'first answer', 'the FIRST answer is the title candidate, not the last'


def test_empty_placeholder_user_message_is_skipped_not_adopted():
    # The resume path writes an empty-text user event. It is not an opener; the real one follows.
    msgs = [
        {'role': 'user', 'content': ''},
        {'role': 'user', 'content': 'the real question'},
        {'role': 'assistant', 'content': 'the answer'},
    ]
    user_text, asst_text = _first_exchange_snippets(msgs)
    assert user_text == 'the real question'
    assert asst_text == 'the answer'


def test_text_opener_behaviour_is_unchanged():
    msgs = [
        {'role': 'user', 'content': 'why is the build red?'},
        {'role': 'assistant', 'content': 'a lint rule fails'},
    ]
    assert _first_exchange_snippets(msgs) == ('why is the build red?', 'a lint rule fails')


def test_latest_exchange_with_image_only_last_turn():
    msgs = [
        {'role': 'user', 'content': 'earlier question'},
        {'role': 'assistant', 'content': 'earlier answer'},
        {'role': 'user', 'content': [_image_part()]},
        {'role': 'assistant', 'content': 'the chart shows a regression'},
    ]
    user_text, asst_text = _latest_exchange_snippets(msgs)
    # `_latest_exchange_snippets` walks backwards and never required user text to collect the
    # assistant side, so it already found the answer; what changed is that the caller accepts it.
    assert asst_text == 'the chart shows a regression'
    assert user_text == 'earlier question'


# --------------------------------------------------------------------- the caller

def _session(messages):
    s = MagicMock()
    s.title = 'Untitled'
    s.llm_title_generated = False
    s.session_id = 'test-image-only-opener'
    s.messages = messages
    return s


def test_regenerate_no_longer_fails_on_an_image_only_opener():
    s = _session([
        {'role': 'user', 'content': [_image_part()]},
        {'role': 'assistant', 'content': 'This is a screenshot of a failing build.'},
    ])
    # `generate_session_title_for_session` does `from api import profiles as profiles_api` inside the
    # function, so the patch target is the module attribute, not a name on api.streaming.
    with patch('api.streaming._aux_title_generation_enabled', return_value=True), \
         patch('api.streaming._generate_llm_session_title_via_aux',
               return_value=('Failing Build Screenshot', 'ok', 'raw')) as mock_aux, \
         patch('api.profiles.profile_env_for_background_worker'):
        title, reason, _raw = generate_session_title_for_session(s)
    assert title == 'Failing Build Screenshot'
    assert reason != 'empty_user_message'
    # The assistant text is what it had to work from; the user side is legitimately empty.
    args, kwargs = mock_aux.call_args
    assert args[0] == ''
    assert args[1] == 'This is a screenshot of a failing build.'


def test_regenerate_still_refuses_a_session_with_nothing_in_it():
    # The reason code is kept for a genuinely untitleable session: no text on either side.
    s = _session([{'role': 'user', 'content': ''}, {'role': 'assistant', 'content': ''}])
    with patch('api.streaming._generate_llm_session_title_via_aux') as mock_aux:
        title, reason, _raw = generate_session_title_for_session(s)
    assert title is None
    assert reason == 'empty_user_message'
    mock_aux.assert_not_called()
