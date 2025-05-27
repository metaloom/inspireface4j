package io.metaloom.inspireface4j;

import io.metaloom.inspireface4j.data.AgeBracket;
import io.metaloom.inspireface4j.data.Gender;
import io.metaloom.inspireface4j.data.Race;

public class FaceAttributes {

	private Race race;
	private AgeBracket age;
	private Gender gender;

	public FaceAttributes(Race race, Gender gender, AgeBracket age) {
		this.race = race;
		this.gender = gender;
		this.age = age;
	}

	public AgeBracket age() {
		return age;
	}

	public Gender gender() {
		return gender;
	}

	public Race race() {
		return race;
	}

	@Override
	public String toString() {
		return "Race: " + race + ", Age: " + age + ", Gender: " + gender;
	}
}
