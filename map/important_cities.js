// Fixed reference cities - these are important locations that don't change
// Distances from these cities will be shown when clicking any location

const fixedCities = [
    {
        name: "Boulder, CO",
        coords: [40.0150, -105.2705],
        description: "Important place",
        short_description: "Sanima House" // Short description for distance labels
    },
    {
        name: "Minneapolis, MN",
        coords: [44.9778, -93.2650],
        description: "Mama Ghar",
        short_description: "Mama Ghar" // Short description for distance labels
    },
    {
        name: "Mt Prospect, IL",
        coords: [42.0664, -87.9373],
        description: "Manisha's place",
        short_description: "Manisha" // Short description for distance labels
    },
    {
        name: "High Point, NC",
        coords: [35.9557, -80.0053],
        description: "Sonisha's place",
        short_description: "Sonisha" // Short description for distance labels
    }
];

/* USAGE INSTRUCTIONS:
-------------------
1. These are fixed reference cities that won't change often
2. Update the description field as needed for each city
3. When you click any city on the map (from cities.js), 
   it will show distances to all these fixed cities
4. Lines on the map will be drawn from New York City to interview locations

CUSTOMIZING DESCRIPTIONS:
You can change "Important place" to anything meaningful:
- "Family home"
- "Current residence"
- "Partner's location"
- "Previous workplace"
- etc.
*/