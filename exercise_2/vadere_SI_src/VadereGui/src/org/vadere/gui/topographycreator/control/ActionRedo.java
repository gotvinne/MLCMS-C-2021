package org.vadere.gui.topographycreator.control;


import org.vadere.util.logging.Logger;

import java.awt.*;
import java.awt.event.ActionEvent;

import javax.swing.*;
import javax.swing.undo.CannotRedoException;
import javax.swing.undo.UndoManager;

/**
 * Action: redo the last action.
 * 
 * 
 */
public class ActionRedo extends AbstractAction {
	private static final long serialVersionUID = 4975524648404524891L;
	private final UndoManager undoManager;
	private final TopographyAction action;
	private static Logger logger = Logger.getLogger(ActionRedo.class);

	public ActionRedo(final String name, final ImageIcon icon, final UndoManager undoManager,
			final TopographyAction action) {
		super(name, icon);
		this.undoManager = undoManager;
		this.action = action;
	}

	@Override
	public void actionPerformed(ActionEvent arg0) {
		try {
			undoManager.redo();
		} catch (CannotRedoException e) {
			logger.debug("Cannot redo! List of edits is empty!");
			Toolkit.getDefaultToolkit().beep();
		}
		action.actionPerformed(arg0);
	}
}
