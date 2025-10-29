from epiclibcpp.epiclib import syllable_counter


def count_total_syllables(text: str) -> int:
    return syllable_counter.count_total_syllables(text)


if __name__ == "__main__":
    # overestimates on 'something'
    for word in "Love Boat promises something for everyone".split(" "):
        print(f"{word} = {count_total_syllables(word)} syllables")
