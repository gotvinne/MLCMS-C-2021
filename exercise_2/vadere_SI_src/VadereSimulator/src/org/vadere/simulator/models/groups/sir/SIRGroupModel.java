package org.vadere.simulator.models.groups.sir;


import org.vadere.annotation.factories.models.ModelClass;
import org.vadere.simulator.context.VadereContext;
import org.vadere.simulator.models.Model;
import org.vadere.simulator.models.groups.AbstractGroupModel;
import org.vadere.simulator.models.groups.Group;
import org.vadere.simulator.models.groups.GroupSizeDeterminator;
import org.vadere.simulator.models.groups.cgm.CentroidGroup;
import org.vadere.simulator.models.potential.fields.IPotentialFieldTarget;
import org.vadere.simulator.projects.Domain;
import org.vadere.state.attributes.Attributes;
import org.vadere.simulator.models.groups.sir.SIRGroup;
import org.vadere.state.attributes.models.AttributesSIRG;
import org.vadere.state.attributes.scenario.AttributesAgent;
import org.vadere.state.scenario.DynamicElementContainer;
import org.vadere.state.scenario.Pedestrian;
import org.vadere.state.scenario.Topography;
import org.vadere.util.geometry.LinkedCellsGrid; //Task 4.4

import java.util.*;

/**
 * Implementation of groups for a susceptible / infected / removed (SIR) model.
 */
@ModelClass
public class SIRGroupModel extends AbstractGroupModel<SIRGroup> {

	private Random random;
	private LinkedHashMap<Integer, SIRGroup> groupsById;
	private Map<Integer, LinkedList<SIRGroup>> sourceNextGroups;
	private AttributesSIRG attributesSIRG;
	private Topography topography;
	private IPotentialFieldTarget potentialFieldTarget;
	private int totalInfected = 0;
	/**
	 * Four lines added to get the context of the variable simTimeStepLength.
	 * This is needed to somewhat decouple the infectionRate and the simTimeStepLength
	 * The variable infectionTimer is used for this purpose
	 */
	private static final String simStepLength = "simTimeStepLength";
	private final int sirUpdateLimit = 5;
	private double simTimeStepLength;
	private double infectionTimer = 0.0;

	public SIRGroupModel() {
		this.groupsById = new LinkedHashMap<>();
		this.sourceNextGroups = new HashMap<>();
	}

	@Override
	public void initialize(List<Attributes> attributesList, Domain domain,
	                       AttributesAgent attributesPedestrian, Random random) {
		this.attributesSIRG = Model.findAttributes(attributesList, AttributesSIRG.class);
		this.topography = domain.getTopography();
		this.random = random;
        this.totalInfected = 0;
        //get the simulation context and initialize the simTimeStepLength var
        this.simTimeStepLength = VadereContext.get(this.topography).getDouble(simStepLength);
	}

	@Override
	public void setPotentialFieldTarget(IPotentialFieldTarget potentialFieldTarget) {
		this.potentialFieldTarget = potentialFieldTarget;
		// update all existing groups
		for (SIRGroup group : groupsById.values()) {
			group.setPotentialFieldTarget(potentialFieldTarget);
		}
	}

	@Override
	public IPotentialFieldTarget getPotentialFieldTarget() {
		return potentialFieldTarget;
	}

	private int getFreeGroupId() {
		if(this.random.nextDouble() < this.attributesSIRG.getInfectionRate()
        || this.totalInfected < this.attributesSIRG.getInfectionsAtStart()) {
			if(!getGroupsById().containsKey(SIRType.ID_INFECTED.ordinal()))
			{
				SIRGroup g = getNewGroup(SIRType.ID_INFECTED.ordinal(), Integer.MAX_VALUE/2);
				getGroupsById().put(SIRType.ID_INFECTED.ordinal(), g);
			}
            this.totalInfected += 1;
			return SIRType.ID_INFECTED.ordinal();
		}
		else{
			if(!getGroupsById().containsKey(SIRType.ID_SUSCEPTIBLE.ordinal()))
			{
				SIRGroup g = getNewGroup(SIRType.ID_SUSCEPTIBLE.ordinal(), Integer.MAX_VALUE/2);
				getGroupsById().put(SIRType.ID_SUSCEPTIBLE.ordinal(), g);
			}
			return SIRType.ID_SUSCEPTIBLE.ordinal();
		}
	}


	@Override
	public void registerGroupSizeDeterminator(int sourceId, GroupSizeDeterminator gsD) {
		sourceNextGroups.put(sourceId, new LinkedList<>());
	}

	@Override
	public int nextGroupForSource(int sourceId) {
		return Integer.MAX_VALUE/2;
	}

	@Override
	public SIRGroup getGroup(final Pedestrian pedestrian) {
		SIRGroup group = groupsById.get(pedestrian.getGroupIds().getFirst());
		assert group != null : "No group found for pedestrian";
		return group;
	}

	@Override
	protected void registerMember(final Pedestrian ped, final SIRGroup group) {
		groupsById.putIfAbsent(ped.getGroupIds().getFirst(), group);
	}

	@Override
	public Map<Integer, SIRGroup> getGroupsById() {
		return groupsById;
	}

	@Override
	protected SIRGroup getNewGroup(final int size) {
		return getNewGroup(getFreeGroupId(), size);
	}

	@Override
	protected SIRGroup getNewGroup(final int id, final int size) {
		if(groupsById.containsKey(id))
		{
			return groupsById.get(id);
		}
		else
		{
			return new SIRGroup(id, this);
		}
	}

	private void initializeGroupsOfInitialPedestrians() {
		// get all pedestrians already in topography
		DynamicElementContainer<Pedestrian> c = topography.getPedestrianDynamicElements();

		if (c.getElements().size() > 0) {
			// TODO: fill in code to assign pedestrians in the scenario at the beginning (i.e., not created by a source)
            //  to INFECTED or SUSCEPTIBLE groups.
		}
	}

	protected void assignToGroup(Pedestrian ped, int groupId) {
		SIRGroup currentGroup = getNewGroup(groupId, Integer.MAX_VALUE/2);
		currentGroup.addMember(ped);
		ped.getGroupIds().clear();
		ped.getGroupSizes().clear();
		ped.addGroupId(currentGroup.getID(), currentGroup.getSize());
		registerMember(ped, currentGroup);
	}

	protected void assignToGroup(Pedestrian ped) {
		int groupId = getFreeGroupId();
		assignToGroup(ped, groupId);
	}


	/* DynamicElement Listeners */

	@Override
	public void elementAdded(Pedestrian pedestrian) {
		assignToGroup(pedestrian);
	}

	@Override
	public void elementRemoved(Pedestrian pedestrian) {
		Group group = groupsById.get(pedestrian.getGroupIds().getFirst());
		if (group.removeMember(pedestrian)) { // if true pedestrian was last member.
			groupsById.remove(group.getID());
		}
	}

	/* Model Interface */

	@Override
	public void preLoop(final double simTimeInSec) {
		initializeGroupsOfInitialPedestrians();
		topography.addElementAddedListener(Pedestrian.class, this);
		topography.addElementRemovedListener(Pedestrian.class, this);
	}

	@Override
	public void postLoop(final double simTimeInSec) {
	}

	@Override
	public void update(final double simTimeInSec) {

		/*
		Implemented task 4.4 optimization of the distance computation using LinkedCellsGrid
		 */

		DynamicElementContainer<Pedestrian> pedContainer = topography.getPedestrianDynamicElements();
		LinkedCellsGrid<Pedestrian> cellsGrid = new LinkedCellsGrid(topography.getAttributes().getBounds().x,
				topography.getAttributes().getBounds().y,
				topography.getAttributes().getBounds().getWidth(),
				topography.getAttributes().getBounds().getHeight(),
				1000);
		if (pedContainer.getElements().size() > 0) {
			for(Pedestrian ped: pedContainer.getElements()){
				cellsGrid.addObject(ped);
			}
			updateSIRGroups(c,cellsGrid);
		}
	}

	/**
	 * Method to update SIR groups. The var infectionTimer is used to ensure that the infectionRate is note directly
	 * dependent on the var simTimeStepLength. i.e if the simTimeStepLength is 0.1 then the if statement will execute 1 time after 50 steps.
	 * Is simTimeStepLength = 0.9 then the if statement will execute after 6 steps and so on.
	 * @param container is a DynamicElementContainer containing pedestrian object, needed to iterate through every pedestrian.
	 * @param cellsGrid is a LinkedCellsGrid. needed to check which pedestrian is closest to the infected source.
	 */
	private void updateSIRGroups(DynamicElementContainer<Pedestrian> container, LinkedCellsGrid<Pedestrian> cellsGrid){
		if (this.infectionTimer >= this.sirUpdateLimit) {
			for (Pedestrian ped : container.getElements()) {
				int groupId = getGroup(ped).getID();
					
				for (Pedestrian ped_neighbour : cellsGrid.getObjects(ped.getPosition(), attributesSIRG.getInfectionMaxDistance())) {
					if (ped == ped_neighbour || getGroup(ped_neighbour).getID() != SIRType.ID_INFECTED.ordinal())
						continue;
					if (this.random.nextDouble() < attributesSIRG.getInfectionRate()) {
						if (groupId == SIRType.ID_SUSCEPTIBLE.ordinal()) {
							elementRemoved(ped);
							assignToGroup(ped, SIRType.ID_INFECTED.ordinal());
						}
					}
				}
			}
			this.infectionTimer = 0.0;
		}
		this.infectionTimer += this.simTimeStepLength;
	}
}