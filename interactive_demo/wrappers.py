from PyQt5 import QtWidgets, QtGui, QtCore


class FocusButton(QtWidgets.QPushButton):
    def __init__(self, *args, highlightthickness=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(QtCore.Qt.ClickFocus)


class FocusLabelFrame(QtWidgets.QGroupBox):
    def __init__(self, *args, highlightthickness=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(QtCore.Qt.ClickFocus)


class BoundedNumericalEntry(QtWidgets.QLineEdit):
    def __init__(self, parent=None, min_value=None, max_value=None, variable=None,
                 vartype=float, width=7, allow_inf=False, **kwargs):
        super().__init__(parent)
        self.validator = QtGui.QDoubleValidator() if vartype == float else QtGui.QIntValidator()
        self.setValidator(self.validator)

        if variable is None:
            if vartype == float:
                self.var = QtCore.QVariant(float())
            elif vartype == int:
                self.var = QtCore.QVariant(int())
            else:
                self.var = QtCore.QVariant(str())
        else:
            self.var = variable

        self.vartype = vartype
        self.old_value = self.var
        self.allow_inf = allow_inf

        self.min_value, self.max_value = min_value, max_value

        self.textChanged.connect(self._check_bounds)

    def _check_bounds(self):
        instr = self.text()
        if self.allow_inf and instr == 'INF':
            self.setText('INF')
            return

        try:
            new_value = self.vartype(instr)
        except ValueError:
            return

        if (self.min_value is None or new_value >= self.min_value) and \
                (self.max_value is None or new_value <= self.max_value):
            if new_value != self.old_value:
                self.old_value = self.vartype(self.text())
                self.setText(str(self.old_value))
                self.var = QtCore.QVariant(self.old_value)
        else:
            self.setText(str(self.old_value))
            mn = '-inf' if self.min_value is None else str(self.min_value)
            mx = '+inf' if self.max_value is None else str(self.max_value)
            message_box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning,
                                                "Incorrect value in input field",
                                                f"Value should be in [{mn}; {mx}] and of type {self.vartype.__name__}",
                                                QtWidgets.QMessageBox.Ok)
            message_box.exec_()


class FocusHorizontalSlider(QtWidgets.QSlider):
    def __init__(self, *args, orientation=QtCore.Qt.Horizontal, slider_position=QtWidgets.QSlider.SliderToMinimum,
                 resolution=1,
                 slider_length=20, length=200, **kwargs):
        super().__init__(orientation)
        self.setSingleStep(resolution)
        self.setSliderPosition(slider_position)
        self.setFixedLength(length)


class FocusCheckBox(QtWidgets.QCheckBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class FocusPushButton(QtWidgets.QPushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class FocusGroupBox(QtWidgets.QGroupBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class FocusHorizontalScale:
    pass