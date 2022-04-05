package org.vadere.gui.topographycreator.control;

import java.awt.event.ActionEvent;

import javax.swing.ImageIcon;

import org.vadere.gui.components.control.IMode;
import org.vadere.gui.topographycreator.model.IDrawPanelModel;

/**
 * Action: Sets the selection mode to ZoomOutMode, so the user can zoom out if he hit the mouse
 * button.
 * 
 * 
 */
public class ActionZoomOut extends TopographyAction {

	private static final long serialVersionUID = 804732305457511954L;
	private final IMode mode;

	public ActionZoomOut(final String name, final ImageIcon icon, final IDrawPanelModel panelModel) {
		super(name, icon, panelModel);
		mode = new ZoomOutMode(panelModel);
	}

	@Override
	public void actionPerformed(ActionEvent arg0) {
		getScenarioPanelModel().setMouseSelectionMode(mode);
		getScenarioPanelModel().notifyObservers();
	}

}
