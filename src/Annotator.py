from tkinter import *
from tkinter import filedialog, Menu, Message, Button, Text, Scrollbar, Frame, Toplevel, Tk, messagebox
from tkinter.font import Font

from functools import partial

from Modules import *
import Tokenizer

import os
import codecs
import json
import re
import time

# Possible colors used for the different tags in the text widget
color_scheme = {"red": "#fc440f", "green": "#99f7ab", "silver": "#c6c7c4", "pearl": "#f1e0c5", "wine": "#592941",
                "deviate": "#2e86ab"}

# Boolean flag to determine, whether additional information should be printed to the console, when auto_annotating
# a given text.
more_info = True

# All the possible tag names already set for the window
annotations = ["SBD", "RightPrediction", "WrongPrediction", "Prediction", "Deviation"]


class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)

        self.master = master
        # activeAnnotation is the tag used when calling the annotation functions
        self.activeAnnotation = "SBD"

        # The text widget used in the window
        self.text = Text(self)

        self.init_window()

        # Data from a loaded text
        self.filepath = ''
        self.directory = ''
        self.filename = ''
        self.annotations = []

        # Pattern is a regex variable used to mark the given pattern in the text widget
        self.pattern = StringVar()
        self.pattern.set(r"[0-9A-Za-zÄäÜüÖöß]*[\\.;\\?\\!:]")

        # Textvariabels for the replace dialog
        self.replace_text = StringVar()
        self.replace_with = StringVar()

        # The annotator used when auto_annotating
        self.annotator = RuleModule()

    def init_window(self):
        """
        Initializes the GUI. Adds all menus and widgets.
        :return:
        """
        self.master.title("Annotator")
        self.pack(fill="both", expand=True)
        self.grid_propagate(False)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        menu = Menu(self.master)
        self.master.config(menu=menu)

        # Teardown menues used for the file dialogs
        file = Menu(menu, tearoff=0)
        file.add_command(label="New", command=self.new)
        file.add_command(label="Load", command=self.load)
        file.add_command(label="Save", command=self.save)
        menu.add_cascade(label="File", menu=file)

        # Teardown menues used for chaning the appearance of the GUI
        """view = Menu(menu, tearoff=0)
        view.add_command(label="Layout", command=self.layout)
        menu.add_cascade(label="View", menu=view)"""

        # Teardown menues for changing the content of the text widget
        edit = Menu(menu, tearoff=0)
        edit.add_command(label="Replace", command=self.replace)
        # edit.add_command(label="Undo", command=self.undo)
        menu.add_cascade(label="Edit", menu=edit)

        # Teardown menues for using the annotation functionality
        annotate = Menu(menu, tearoff=0)
        annotate.add_command(label="Pattern", command=self.define_pattern)
        annotate.add_command(label="Annotate", command=self.auto_annotate)
        menu.add_cascade(label="Annotate", menu=annotate)

        # Teardown menues for changing the annotation type
        annotators = Menu(menu, tearoff=0)
        annotators.add_command(label="Rule", command=self.set_rule)
        annotators.add_command(label="Template", command=self.set_template)
        annotators.add_command(label="Punkt", command=self.set_punkt)
        annotators.add_command(label="CRF Law", command=self.set_crf_law)
        annotators.add_command(label="CRF Jug", command=self.set_crf_jug)
        annotators.add_command(label="NN Law", command=self.set_nn_law)
        annotators.add_command(label="NN Jug", command=self.set_nn_jug)
        annotators.add_command(label="OpenNLP", command=self.set_opennlp)
        menu.add_cascade(label="Annotators", menu=annotators)

        # Building the text widgets with its corresponding tags
        self.text.configure(font=font)
        self.text.grid(columnspan=20, rowspan=60, sticky=N+E+S+W)
        self.text.tag_config("SBD", background=color_scheme["red"])
        self.text.tag_config("Pattern", background=color_scheme["silver"])
        self.text.tag_config("RightPrediction", background=color_scheme["green"])
        self.text.tag_config("WrongPrediction", background=color_scheme["wine"])
        self.text.tag_config("Prediction", background=color_scheme["pearl"])
        self.text.tag_config("Deviation", background=color_scheme["deviate"])
        self.text.tag_raise("Deviation")
        self.text.tag_raise("SBD")
        self.text.tag_raise("RightPrediction")
        self.text.tag_raise("WrongPrediction")
        self.text.tag_lower("Pattern")

        # Initializing the scrollbar
        scrollb = Scrollbar(self, command=self.text.yview)
        self.text['yscrollcommand'] = scrollb.set
        scrollb.grid(column=20, row=0, rowspan=60, sticky=W+E+N+S)

        self.text.bind("<ButtonRelease-1>", self.quick_annotate)
        self.text.bind("<Alt-ButtonRelease-1>", self.exact_annotate)
        self.text.bind("<Shift-ButtonRelease-1>", self.del_annotation)

    def new(self):
        """
        Action performed in the file menu. Starts the processes for changing the current file information.
        If the text widget is filled with content, then an Save Dialog is started.
        :return: None
        """
        output = self.text.get("1.0", END)
        if len(output) > 1:
            self.popup_save_dialog()
        else:
            self.start_new()

    def load(self):
        """
        Action performed in the file menu. Opens up a file dialog and then starts the processes to load different
        file types. Possible file extensions are .txt and .json
        :return: None
        """
        if len(self.directory) <= 0:
            self.filepath = os.getcwd()
        tempdir = filedialog.askopenfilename(parent=self.master, initialdir=self.directory, title="Create new document",
                                             filetypes=(("JSON Files", "*.json"), ("Text Files", "*.txt"), ("all files", "*.*")))

        if len(tempdir) > 0:
            self.filepath = tempdir
            (head, file) = os.path.split(tempdir)
            self.directory = head
            self.filename = file
            if self.filename.endswith(".txt"):
                self.load_text()
            elif self.filename.endswith(".json"):
                self.load_json()

    def save(self):
        """
        Writes the data to the given file location. Annotations are saved in a .json file with the plaintext.
        The plaintext is also saved in a .txt file.
        :return: None
        """
        output = self.text.get("1.0", END)
        if self.filename == '':
            self.start_new()
        # Saving to the .txt file
        if self.filename.endswith(".json"):
            old_filename = self.filename
            self.filename = self.filename.replace(".json", ".txt")
            self.filepath = self.filepath.replace(old_filename, self.filename)
        with codecs.open(self.filepath, 'w+', 'utf-8') as f:
            f.write(output)
        # Saving to the .json file
        dic = {"Text": output, "Annotations": self.annotations}
        json_filepath = self.filepath.replace(".txt", ".json")
        with codecs.open(json_filepath, "w+", 'utf-8') as f:
            json.dump(dic, f, sort_keys=True, indent=4, ensure_ascii=False)

    def layout(self):
        """
        An action performed from the view menu. Starts a dialog to change the style of the given window.
        :return: None
        """
        raise NotImplementedError

    def replace(self):
        """
        An action performed from the edit menu. Starts a dialog to replace certain parts of the text being processed.
        :return:
        """
        current_text = self.text.get("1.0", END)
        replace_dialog = Toplevel()
        x, y = self.dialog_position(replace_dialog)
        replace_dialog.geometry("+%d+%d" % (x, y))
        replace_dialog.title("Replacement")
        replace_entry = Entry(replace_dialog, textvariable=self.replace_text)
        replace_entry.grid(column=1, row=1)
        replace_with_entry = Entry(replace_dialog, textvariable=self.replace_with)
        replace_with_entry.grid(column=2, row=1)
        replace_button = Button(replace_dialog, text="Replace", command=self.process_replace)
        replace_button.grid(column=2, row=2)

    def undo(self):
        """
        An action performed from the edit menu. Undos the last changes to the document
        :return:
        """
        raise NotImplementedError

    def load_text(self):
        """
        Loads the text given in the filepath into the text widget. A .txt file is assumed.
        :return: None
        """
        self.text.delete('1.0', END)
        self.annotations = []
        with codecs.open(self.filepath, 'r', 'utf-8') as f:
            text = f.read()
        self.text.insert(END, text)

    def load_json(self):
        """
        Loads the text given in the filepath into the text widget. A .json file is assumed.
        :return: None
        """
        self.text.delete('1.0', END)
        self.annotations = []
        with codecs.open(self.filepath, 'r', 'utf-8') as f:
            dic = json.load(f)
        self.text.insert(END, dic['Text'])
        self.annotations = dic['Annotations']
        self.reannotate()

    def define_pattern(self):
        """
        Starts a dialog window to define an annotation pattern.
        :return: None
        """
        pattern_dialog = Toplevel()
        pattern_dialog.resizable(False, False)
        px, py = self.dialog_position(pattern_dialog)
        pattern_dialog.geometry("250x50+%d+%d" % (px, py))
        pattern_dialog.title("Pattern")
        pattern_entry = Entry(pattern_dialog, textvariable=self.pattern)
        pattern_entry.pack()
        pattern_button = Button(pattern_dialog, text="Ok", command=self.process_pattern)
        pattern_button.pack()

    def process_pattern(self):
        """
        Marks all occurences of the defined pattern. Pattern is saved in the self.pattern entry.
        :return:
        """
        self.text.tag_remove("Pattern", "1.0", END)
        sentences = self.text.get("1.0", END).split("\n")
        pattern = self.pattern.get()
        pattern = re.compile(pattern)
        for i, sentence in enumerate(sentences, start=1):
            for m in pattern.finditer(sentence):
                pat_start = str(i) + "." + str(m.start())
                pat_end = str(i) + "." + str(m.end())
                self.text.mark_set("matchStart", pat_start)
                self.text.mark_set("matchEnd", pat_end)
                self.text.tag_add("Pattern", "matchStart", "matchEnd")

    def process_replace(self, pattern=None, replacement=None):
        """
        Finds all occurences in the text of the exact pattern. Replaces it with the given text.
        Labels will be recalculated and reset. Active predictions will be removed.
        """
        if pattern is None:
            pattern = self.replace_text.get()
        if replacement is None:
            replacement = self.replace_with.get()
        text = self.text.get("1.0", END)
        len_pattern = len(pattern)
        len_replacement = len(replacement)
        displacement = len_replacement - len_pattern

    def start_new(self):
        """
        Opens up Filedialog, to set the location of the current text file.
        Resets the current information about the working file.
        :return: None
        """
        if len(self.directory) <= 0:
            self.directory = os.getcwd()
        tempdir = filedialog.asksaveasfilename(parent=self.master, initialdir=self.directory, title="Create new document",
                                               filetypes=(("JSON Files", "*.json"), ("Text Files", "*.txt"), ("all files", "*.*")))
        if len(tempdir) > 0:
            self.filepath = tempdir
            (head, file) = os.path.split(tempdir)
            self.directory = head
            self.filename = file
            self.annotations = []

    def popup_save_dialog(self):
        """
        Starts the save dialog, used when creating a "new" file. Allows the possibility to Save the current text or
        dismiss all data.
        :return:
        """
        save_dialog = Toplevel()
        save_dialog.resizable(False, False)
        px, py = self.dialog_position(save_dialog)
        save_dialog.geometry("+%d+%d" % (px, py))
        save_dialog.title("Save")
        save_dialog.transient(self)

        msg = Message(save_dialog, text="Save your current progress?")
        msg.grid(row=0, columnspan=2)
        save = Button(save_dialog, text="Save", command=self.save_dialog_opt(save_dialog))
        save.grid(row=1, column=0, sticky=N+W+S+E, padx=2)
        dismiss = Button(save_dialog, text="Dismiss", command=self.destroy_dialog_opt(save_dialog))
        dismiss.grid(row=1, column=1, sticky=N+W+S+E, padx=2)

    def save_dialog_opt(self, save_dialog):
        """
        An action called from the save dialog window. Saves the current progress and then starts a new text.
        :param save_dialog: The save dialog assigned with this action.
        :return: None
        """
        save_dialog.destroy()
        self.save()
        self.start_new()

    def destroy_dialog_opt(self, save_dialog):
        """
        An action called form the save dialog window. Does not save the current progress and starts a new text.
        :param save_dialog: The save dialog assigned with this action.
        :return: None
        """
        save_dialog.destroy()
        self.start_new()

    def reannotate(self):
        """
        Annotates the text in the widget with the saved annotations. This is needed after reloading a given file.
        :return: None
        """
        sentences = self.text.get("1.0", END).split("\n")
        self.annotations.sort(key=get_start)
        running_position = 0
        running_line = 1
        anno_iter = iter(self.annotations)
        try:
            annotation = next(anno_iter)
        except StopIteration:
            annotation = None
        start_index = None
        end_index = None
        for sentence in sentences:
            if annotation is None:
                break
            # Because we omit the newline at the end; different length
            length = len(sentence)+1
            while annotation['start'] < running_position+length:
                if annotation['start'] < running_position+length and start_index is None:
                    start_index = str(running_line)+"."+str(annotation['start']-running_position)
                if annotation['end'] < running_position+length and end_index is None:
                    end_index = str(running_line)+"."+str(annotation['end']-running_position)
                if start_index is not None and end_index is not None:
                    self.text.tag_add(annotation['type'], start_index, end_index)
                    start_index = None
                    end_index = None
                    try:
                        annotation = next(anno_iter)
                    except StopIteration:
                        annotation = None
                        break
            running_line = running_line + 1
            running_position = running_position + length

    def quick_annotate(self, event):
        """
        Action performed to annotate a word. When calling this method the current selection selected for annotation.
        If the current selection ends with a special character, it will be removed from the selection.
        This is needed to quickly annotate a large number of documents.
        :param event: The event calling this method.
        :return:
        """
        if self.text.tag_ranges(SEL):
            s_count = self.text.count("1.0", SEL_FIRST)
            if s_count is None:
                start = 0
            else:
                start = s_count[0]
            e_count = self.text.count("1.0", SEL_LAST)
            if e_count is None:
                end = 0
            else:
                end = e_count[0]
            start_index, end_index = (self.text.index(SEL_FIRST), self.text.index(SEL_LAST))
            annotation = self.text.get(SEL_FIRST, SEL_LAST)
            if annotation.endswith(('.', ',', '!', '?', '\"', ':', ';')):
                end = end-1
                split = end_index.split('.')
                if len(split) == 2:
                    line, character = split[0], split[1]
                    character = int(character)-1
                    end_index = line + "." + str(character)
                else:
                    raise ValueError("Wrong line format, while annotating.")
                self.text.tag_remove(SEL, "1.0", END)
                self.text.tag_add(SEL, start_index, end_index)
                self.text.focus_force()
                try:
                    annotation = self.text.get(SEL_FIRST, SEL_LAST)
                    self.annotate(annotation, start, end, start_index, end_index)
                except TclError:
                    messagebox.showerror("No word selected...")
            else:
                self.annotate(annotation, start, end, start_index, end_index)

    def exact_annotate(self, event):
        """
        Action performed to annotate a word. When calling this method the current selection selected for annotation.
        :param event: The event calling this method.
        :return:
        """
        if self.text.tag_ranges(SEL):
            start, end = (self.text.count("1.0", SEL_FIRST)[0], self.text.count("1.0", SEL_LAST)[0])
            start_index, end_index = (self.text.index(SEL_FIRST), self.text.index(SEL_LAST))
            annotation = self.text.get(SEL_FIRST, SEL_LAST)
            self.annotate(annotation, start, end, start_index, end_index)

    def annotate(self, anno, s, e, s_index, e_index):
        """
        Adds a annotation to the annotation list and visually marks the token in the text widget.
        :param anno: The text of the annotation, i.e. the token.
        :param s: The index of the first character of the token.
        :param e: The index of the end of the token. This is the first character not included in the annotation.
        :param s_index: The index of the first character in "Line.Character" format
        :param e_index: The index of the end of the token in "Line.Character" format
        :return: None
        """
        self.text.tag_add(self.activeAnnotation, s_index, e_index)
        self.annotations.append({"annotation": anno, "start": s, "end": e, "type": self.activeAnnotation})

    def del_annotation(self, event):
        """
        Deletes all given tags of the self.activeAnnotation type in the selected region.
        :param event:
        :return:
        """
        if self.text.tag_ranges(SEL):
            start, end = (self.text.count("1.0", SEL_FIRST)[0], self.text.count("1.0", SEL_LAST)[0])
            start_index, end_index = (self.text.index(SEL_FIRST), self.text.index(SEL_LAST))
            self.text.tag_remove(self.activeAnnotation, start_index, end_index)
            self.annotations = [anno for anno in self.annotations
                                if anno["start"] < start or anno["end"] > end or anno["type"] != self.activeAnnotation]

    def auto_annotate(self):
        """
        Calculate sentence boundary annotations with the given SBDModule. These annotations are then plotted against
        the groundtruth or marked as the prediction.
        :return: None
        """
        text = self.text.get("1.0", END)
        if len(self.annotations) > 0:
            tokens, labels = Tokenizer.tokenize_json({"Text": text, "Annotations": self.annotations})
        else:
            tokens = Tokenizer.tokenize_text(text)
            labels = None
        if self.annotator.predict_on_token:
            predictions = self.annotator.predict(tokens)
        else:
            predictions = self.annotator.predict_text(text)
        if labels is None:
            self.text.tag_remove("Prediction", "1.0", END)
            self.activeAnnotation = "Prediction"
            self.annotate_tokens(predictions)
        else:
            if more_info:
                print(self.filename, calculateperformance(labels, predictions))
            true_prediction = []
            false_prediction = []
            deviation = []
            for i, (true_label, predicted_label) in enumerate(zip(labels, predictions)):
                if true_label:
                    if predicted_label:
                        true_prediction.append(True)
                        false_prediction.append(False)
                        deviation.append(False)
                    else:
                        true_prediction.append(False)
                        false_prediction.append(False)
                        deviation.append(True)
                else:
                    if predicted_label:
                        true_prediction.append(False)
                        false_prediction.append(True)
                        deviation.append(True)
                    else:
                        true_prediction.append(False)
                        false_prediction.append(False)
                        deviation.append(False)
            self.text.tag_remove("Deviation", "1.0", END)
            self.activeAnnotation = "Deviation"
            self.annotate_tokens(deviation)
            self.text.tag_remove("RightPrediction", "1.0", END)
            self.activeAnnotation = "RightPrediction"
            self.annotate_tokens(true_prediction)
            self.text.tag_remove("WrongPrediction", "1.0", END)
            self.activeAnnotation = "WrongPrediction"
            self.annotate_tokens(false_prediction)
        self.activeAnnotation = "SBD"

    def annotate_tokens(self, values):
        """
        Annotates the text with self.activeAnnotation given by the truth values for every individual token.
        The logic of the Tokenizer module is used to extract the tokens.
        :param values: List of truth values for every token in the text, whether it should be annotated
        :return: None
        """
        text = self.text.get("1.0", END)
        index = 0
        character = 0
        token_start = 0
        none_space = False
        last_annotated = False
        line = 1
        for c in text:
            if re.match(Tokenizer.alphanumeric, c):
                character += 1
                none_space = True
                last_annotated = False
            else:
                # Annotate last token if necessary, i.e. none space
                if none_space and not last_annotated:
                    token_end = character
                    str_start = str(line)+"."+str(token_start)
                    str_end = str(line)+"."+str(token_end)
                    if values[index]:
                        self.text.tag_add(self.activeAnnotation, str_start, str_end)
                    index += 1
                # Look at new token
                token_start = character
                character += 1
                if c in Tokenizer.filter_character:
                    none_space = False
                else:
                    none_space = True
                    last_annotated = True
                    token_end = character
                    str_start = str(line) + "." + str(token_start)
                    str_end = str(line) + "." + str(token_end)
                    if index < len(values) and values[index]:
                        self.text.tag_add(self.activeAnnotation, str_start, str_end)
                    index += 1
                if c == "\n":
                    line += 1
                    character = 0
                token_start = character

    def set_rule(self):
        """
        Sets the active annotator to the Rule Module
        """
        self.annotator = RuleModule()

    def set_template(self):
        """
        Sets the annotator to the template module. Uses the last selected annotator in a combined approach.
        """
        if not isinstance(self.annotator, TemplateModule):
            self.annotator = TemplateModule(True, self.annotator)

    def set_punkt(self):
        """
        Sets the active annotator to the Punkt Module
        """
        self.annotator = NLTKModule()

    def set_crf_law(self):
        """
        Sets the active annotator to the CRF Module
        """
        path = "ges_best_len(7,7)_sig(5,5)_lower(7,7)_spec(10,10)_islow(3,3)_isup(3,3)_isnum(3,3)"
        dir = "Models"
        print("CRF: ", dir, path)
        self.annotator = CRFModule(save_directory=dir, model_path=path,
                                   feat_ex=[LengthExtractor(7, 7), SignatureExtractor(5, 5), LowercaseExtractor(7, 7),
                                            LowerExtractor(3, 3), UpperExtractor(3, 3), NumberExtractor(3, 3),
                                            SpecialExtractor(10, 10)])

    def set_crf_jug(self):
        """
        Sets the active annotator to the CRF Module
        """
        path = "jug_best_len(5,5)_sig(5,5)_spec(5,5)_lower(5,5)_islow(3,3)_isup(3,3)_isnum(3,3)"
        dir = "Models"
        print("CRF: ", dir, path)
        self.annotator = CRFModule(save_directory=dir, model_path=path,
                                   feat_ex=[LengthExtractor(5, 5), SignatureExtractor(5, 5), LowercaseExtractor(5, 5),
                                            LowerExtractor(3, 3), UpperExtractor(3, 3), NumberExtractor(3, 3),
                                            SpecialExtractor(5, 5)])

    def set_nn_law(self):
        """
        Sets the active annotator to the CRF Module
        """
        path = "ges_lstm_24.model"
        dir = "Models"
        print("NN: ", dir, path)
        self.annotator = NNModule(vec_model_file="word2vec.wv", model_file=path, self_trained_embeddings=True,
                                  embedding_size=100)

    def set_nn_jug(self):
        """
        Sets the active annotator to the CRF Module
        """
        path = "jug_gru_22.model"
        dir = "Models"
        print("NN: ", dir, path)
        self.annotator = NNModule(vec_model_file="word2vec.wv", model_file=path, self_trained_embeddings=True,
                                  embedding_size=100)

    def set_opennlp(self):
        """
        Sets the active annotator to the OpenNLP Module
        """
        path = "jug_open_nlp.bin"
        self.annotator = OpenNLPModule(model_file=path)

    def dialog_position(self, dialog):
        """
        Calculates the position for the dialog window. Assigns a position in the middle of the screen.
        :param dialog: Dialog window
        :return: (x, y) position for the window
        """
        dw = dialog.winfo_width()
        dh = dialog.winfo_height()
        ww = self.winfo_width()
        wh = self.winfo_height()
        wx = self.master.winfo_x()
        wy = self.master.winfo_y()
        w_midx = wx + (ww/2)
        w_midy = wy + (wh/2)
        w_midx = w_midx - (dw/2)
        w_midy = w_midy - (dh/2)
        return w_midx, w_midy


def get_start(annotation):
    """
    Method returns the start of the annotation. Used for sorting the annotations before reannotating.
    :param annotation: Annotation
    :return: Start of the annotation
    """
    return annotation['start']


root = Tk()

font = Font(family="bitstream charter", size=13)

width = 875
height = 800
ws = root.winfo_screenwidth()
hs = root.winfo_screenheight()
x = (ws/2) - (width/2)
y = (hs/2) - (height/2)
root.geometry("%dx%d+%d+%d" % (width, height, x, y))

app = Window(root)
app.filepath = os.path.join("../data", "jug_1_AR_30_19.json")
app.load_json()
root.mainloop()
