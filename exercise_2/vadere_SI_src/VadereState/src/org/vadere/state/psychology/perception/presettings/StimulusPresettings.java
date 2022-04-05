package org.vadere.state.psychology.perception.presettings;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.vadere.state.psychology.perception.json.ReactionProbability;
import org.vadere.state.psychology.perception.json.StimulusInfo;
import org.vadere.state.psychology.perception.json.StimulusInfoStore;
import org.vadere.state.psychology.perception.types.*;
import org.vadere.state.util.JacksonObjectMapper;
import org.vadere.util.geometry.shapes.VRectangle;

import java.io.IOException;
import java.util.*;

/**
 * Provide JSON presettings for commonly used stimuli.
 *
 * This class can be used as helper for GUI elements.
 */
public class StimulusPresettings {
    /** Map an event class (e.g., Threat) to a JSON string. */
    public static Map<Class, String> PRESETTINGS_MAP;

    // Static initializer for "PRESETTINGS_MAP".
    static {
        PRESETTINGS_MAP = new HashMap<>();

        Stimulus[] stimuliToUse = new Stimulus[] {
                new Threat(),
                new Wait(),
                new WaitInArea(0, new VRectangle(0, 0, 10, 10)),
                new ChangeTarget(),
                new ChangeTargetScripted()
        };

        for (Stimulus stimulus : stimuliToUse) {
            // Container for a timeframe and the corresponding stimuli.
            StimulusInfo stimulusInfo = new StimulusInfo();

            List<Stimulus> stimuli = new ArrayList<>();
            stimuli.add(stimulus);

            stimulusInfo.setTimeframe(new Timeframe(0, 10, false, 0));
            stimulusInfo.setStimuli(stimuli);

            // Container for multiple stimulus infos.
            List<StimulusInfo> stimulusInfos = new ArrayList<>();
            stimulusInfos.add(stimulusInfo);

            StimulusInfoStore stimulusInfoStore = new StimulusInfoStore();
            stimulusInfoStore.setStimulusInfos(stimulusInfos);
            stimulusInfoStore.setReactionProbabilities(Collections.singletonList(new ReactionProbability()));


            try {
                ObjectMapper mapper = new JacksonObjectMapper();
                // String jsonDataString = mapper.writeValueAsString(stimulusInfoStore);
                String jsonDataString = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(stimulusInfoStore);

                PRESETTINGS_MAP.put(stimulus.getClass(), jsonDataString);
            } catch (IOException e) {
                e.printStackTrace();
            }

        }
    }
}
