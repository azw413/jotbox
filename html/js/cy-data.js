
let num_notes = 0;
let num_subjects = 0;

function graph_to_cy(graph)
{
    let elements = [];
    let links = new Map();

    let defaultColours = [ '#e5e1d6', '#d5d1d6', '#c5c1d6', '#b5b1d6', '#a5a1d6', '#d5e1c6', '#c5e1b6', '#b5e1a6', '#e5d1c6' ];
    const palette = window.TOPIC_COLOUR_PALETTE || defaultColours;

    num_notes = graph.nodes.length;
    num_subjects = 0;

    graph.nodes.forEach(function(n) {

        if (n.group > num_subjects) { num_subjects = n.group; }
        const colour = window.topicColorFor ? window.topicColorFor(n.topic || '', n.group) : palette[n.group % palette.length];

        let node = {
            group: "nodes",
            data: {
                id: n.id,
                label: n.name,
                topic: n.topic || '',
                color: colour
            },
            grabbable: false
        }
        elements.push(node);
    });

    num_subjects = num_subjects + 1;

    graph.links.forEach(function(l) {
        if (l.source !== l.target) {

            let key = {s: l.source, t: l.target};
            if (key.s > key.t) key = {s: l.target, t: l.source};

            let key_string = JSON.stringify(key);

            if (!links.has(key_string))
            {
                links.set(key_string, 1);
            }
            else
            {
                links.set(key_string, 2);
            }
        }
    });

    for (let [key_string, value] of links) {
          let key = JSON.parse(key_string);
          let edge = {
               group: "edges",
               data: {
                   source: key.s,
                   target: key.t
               },
              style: {
                  'width': 1 + value,
              },
              selectable: false
           };
           elements.push(edge);
    }

    return elements;
}
