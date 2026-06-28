#let week-rule = line(length: 100%, stroke: 1pt)
#let day-rule = line(length: 100%, stroke: 0.25pt)

// Ordinal suffix helper
#let ordinal(n) = {
  let s = str(n)
  if n == 11 or n == 12 or n == 13 { s + "th" }
  else if calc.rem(n, 10) == 1 { s + "st" }
  else if calc.rem(n, 10) == 2 { s + "nd" }
  else if calc.rem(n, 10) == 3 { s + "rd" }
  else { s + "th" }
}

#let month-name(m) = (
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December"
).at(m - 1)

#let weekday-name(w) = (
  "Sunday", "Monday", "Tuesday", "Wednesday",
  "Thursday", "Friday", "Saturday"
).at(w)

#let weekday(year, month, day) = {
  let m = month
  let y = year
  if m < 3 { m += 12; y -= 1 }
  let k = calc.rem(y, 100)
  let j = int(y / 100)
  let h = calc.rem(
    day + int((13 * (m + 1)) / 5) + k + int(k / 4) + int(j / 4) + 5 * j,
    7
  )
  calc.rem(h + 6, 7)
}

#let days-in-month(year, month) = {
  if month == 2 {
    let leap = calc.rem(year, 4) == 0 and (
      calc.rem(year, 100) != 0 or calc.rem(year, 400) == 0
    )
    if leap { 29 } else { 28 }
  } else if month in (4, 6, 9, 11) { 30 }
  else { 31 }
}

#let next-day(year, month, day) = {
  let d = day + 1
  if d > days-in-month(year, month) {
    d = 1
    let m = month + 1
    if m > 12 { (year + 1, 1, d) }
    else { (year, m, d) }
  } else {
    (year, month, d)
  }
}

#let fmt-month-day(month, day) = {
  month-name(month) + " " + ordinal(day)
}

#let fmt-date(year, month, day) = {
  let wd = weekday-name(weekday(year, month, day))
  wd + ", " + month-name(month) + " " + ordinal(day)
}

// Emits a week heading + 7 day headings, with body content interleaved.
//
// Parameters:
//   start  – (year, month, day) tuple for the Monday of the week
//   bodies – array of 7 content blocks (Mon→Sun)
//
#let week(start, intro: [], bodies) = {
  week-rule
  let (sy, sm, sd) = start
  [== Week of #fmt-month-day(sm, sd)]
  intro
  let cur = (sy, sm, sd)
  for i in range(7) {
    let (cy, cm, cd) = cur
    day-rule
    [=== #fmt-date(cy, cm, cd)]
    bodies.at(i)
    if i < 6 { cur = next-day(cy, cm, cd) }
  }
}

#week((2026, 4, 27), intro: [], (
  [/* Monday */],
  [/* Tuesday */],
  [/* Wednesday */],
  [/* Thursday */],
  [/* Friday */],
  [/* Saturday */],
  [/* Sunday */],
))

#week((2026, 5, 4), intro: [], (
  [/* Monday */],
  [/* Tuesday */],
  [/* Wednesday */],
  [/* Thursday */],
  [/* Friday */],
  [/* Saturday */],
  [/* Sunday */],
))

#week((2026, 5, 11), intro: [], (
  [/* Monday */],
  [/* Tuesday */],
  [/* Wednesday */],
  [/* Thursday */],
  [/* Friday */],
  [/* Saturday */],
  [/* Sunday */],
))

#week((2026, 5, 18), intro: [], (
  [/* Monday */],
  [/* Tuesday */],
  [/* Wednesday */],
  [/* Thursday */],
  [/* Friday */],
  [/* Saturday */],
  [/* Sunday */],
))

#week((2026, 5, 25), intro: [], (
  [/* Monday */],
  [/* Tuesday */],
  [/* Wednesday */],
  [/* Thursday */],
  [/* Friday */],
  [/* Saturday */],
  [/* Sunday */],
))

#week((2026, 6, 1), intro: [], (
  [/* Monday */],
  [/* Tuesday */],
  [/* Wednesday */],
  [/* Thursday */],
  [/* Friday */],
  [/* Saturday */],
  [/* Sunday */],
))

#week((2026, 6, 8), intro: [], (
  [/* Monday */],
  [/* Tuesday */],
  [/* Wednesday */],
  [/* Thursday */],
  [/* Friday */],
  [/* Saturday */],
  [/* Sunday */],
))

#week((2026, 6, 15), intro: [], (
  [/* Monday */],
  [/* Tuesday */],
  [/* Wednesday */],
  [/* Thursday */],
  [/* Friday */],
  [/* Saturday */],
  [/* Sunday */],
))

#week((2026, 6, 22), intro: [], (
  [/* Monday */],
  [/* Tuesday */],
  [/* Wednesday */],
  [/* Thursday */],
  [/* Friday */],
  [/* Saturday */],
  [/* Sunday */],
))

#week((2026, 6, 29), intro: [], (
  [/* Monday */],
  [/* Tuesday */],
  [/* Wednesday */],
  [/* Thursday */],
  [/* Friday */],
  [/* Saturday */],
  [/* Sunday */],
))

#week((2026, 7, 6), intro: [], (
  [/* Monday */],
  [/* Tuesday */],
  [/* Wednesday */],
  [/* Thursday */],
  [/* Friday */],
  [/* Saturday */],
  [/* Sunday */],
))

#week((2026, 7, 13), intro: [], (
  [/* Monday */],
  [/* Tuesday */],
  [/* Wednesday */],
  [/* Thursday */],
  [/* Friday */],
  [/* Saturday */],
  [/* Sunday */],
))

#week((2026, 7, 20), intro: [], (
  [/* Monday */],
  [/* Tuesday */],
  [/* Wednesday */],
  [/* Thursday */],
  [/* Friday */],
  [/* Saturday */],
  [/* Sunday */],
))

#week((2026, 7, 27), intro: [], (
  [/* Monday */],
  [/* Tuesday */],
  [/* Wednesday */],
  [/* Thursday */],
  [/* Friday */],
  [/* Saturday */],
  [/* Sunday */],
))