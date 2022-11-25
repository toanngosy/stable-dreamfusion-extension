function start_training_dreambooth(){
    requestProgress('df');
    gradioApp().querySelector('#df_error').innerHTML='';
    gradioApp().querySelector('#df_status').innerHTML='';
    return args_to_array(arguments);
}

onUiUpdate(function(){
    check_progressbar('df', 'df_progressbar', 'df_progress_span', '', 'df_interrupt', 'df_preview', 'df_gallery')
});

ex_titles = titles;

console.log("Existing titles: ", ex_titles);

new_titles = {
	"Train": "Start training.",
	"Cancel": "Cancel training."
}

