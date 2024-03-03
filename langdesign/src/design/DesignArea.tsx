import { useState, useContext, useRef } from "react"
import { ImageUploader } from "./ImageUploader";
import { FaPaperPlane } from 'react-icons/fa'; 
import { DesignContext } from "./DesignProvider";
import { getLogger } from "../core";
import { FaArrowRotateLeft, FaArrowRotateRight, FaX, FaDownload, FaRegPenToSquare, FaCheck } from "react-icons/fa6";
import { saveAs } from 'file-saver';

import './css/design-area.css'
import { AuthContext } from "../auth/AuthProvider";
import { HistoryDesign } from "./HistoryDesign";
import { ForwardDesignCard } from "./ForwardDesignCard";

const log = getLogger('DesignArea');

export const DesignArea: React.FC = () => {
    const [promptText, setPromptText] = useState('');
    const [editedName, setEditedName] = useState('');
    const [isNameEditing, setIsNameEditing] = useState(false);
    const [isNextVisible, setIsNextVisible] = useState(false)
    const designTitleInputRef = useRef<HTMLInputElement>(null);
    const { upload, design, undo, loadDesign, startNew, forward, rename, currentDesign, designs } = useContext(DesignContext);
    const { uid } = useContext(AuthContext);

    function handleImageUpload(image: File) {
        log('File uploaded')
        //setUploadedImage(file)
        upload && upload(image, uid || "")
            .then(url => {
            })
            .catch(error => {
        });
    }

    function handlePromptSend() {
        log('Design started')
        design && currentDesign && currentDesign.images && design(currentDesign?.images[currentDesign.currentImageIndex].imageUrl, promptText, uid || "")
            .then(url => {
                setPromptText('')
            })
            .catch(error => {
        });
    }

    function handleUndo() {
        log('Undo')
        undo && undo();
    }

    function handleStartNewDesign() {
        log('Start new design')
        startNew && startNew();
    }

    function handleForwardClose(imageUrl: string) {
        if(!forward) {
            return;
        }
        forward(imageUrl);
        setIsNextVisible(false);
    }

    function forwardDisabled() {
        return currentDesign && currentDesign.images && 
        (currentDesign.images[currentDesign.currentImageIndex].children == undefined || currentDesign.images[currentDesign.currentImageIndex].children?.length == 0)
    }

    function handleSaveImage() {
        if (currentDesign && currentDesign.images) {
            const imageUrl = currentDesign.images[currentDesign.currentImageIndex].imageUrl;
            fetch(imageUrl)
                .then(response => response.blob())
                .then(blob => {
                    saveAs(blob, `${currentDesign.name}.png`);
                })
                .catch(error => {
                    console.error('Error saving image:', error);
                });
        }
    }

    function handleRename() {
        if(!isNameEditing) {
            setEditedName(currentDesign.name);
            setIsNameEditing(true);
            if(designTitleInputRef.current) {
                console.log('focus');
            }
            designTitleInputRef.current && designTitleInputRef.current.focus();
            return;
        }
        rename && rename(editedName, currentDesign.code);
        setIsNameEditing(false);
    }

    return (
        <div className="design-area">
            {!currentDesign.images && 
                <div className="start-designing">
                    <h1 className="text-design">Start designing</h1>
                    <ImageUploader handleImage={handleImageUpload} />
                </div>
            }
            <div className="desings-history-area">
                    <button id="start-new-button" onClick={handleStartNewDesign} disabled={!currentDesign.images}>Start new design</button>
                    {designs && designs.map(design => (
                        <HistoryDesign key={design.code} name={design.name || "New design"} onClick={() => {
                                loadDesign && loadDesign(design.code, design.name)
                                loadDesign && setIsNameEditing(false)
                                loadDesign && setIsNextVisible(false)
                            }} 
                        />
                        //<p key={design.name} className="design-list-element" onClick={() => loadDesign && loadDesign(design.name)}>{design.name.split("_").at(0)}</p>
                    ))}
            </div>
            {currentDesign.images && 
                <div className='current-design-area'>
                    <div className="toolbar-design-area">
                        <div className="toolbar-title">
                            {!isNameEditing ? 
                                <div className="design-title"><h1 id="design-title">{currentDesign.name}</h1></div>
                                :
                                    <input 
                                        id="design-title-input" 
                                        ref={designTitleInputRef} 
                                        value={editedName} 
                                        type="text" 
                                        onChange={(e: any) => setEditedName(e.target.value)}
                                        onKeyDown={(e) => {
                                            if(e.key === 'Enter') {
                                                handleRename();
                                            }
                                        }}
                                    />                            
                            }
                            <button className="toolbar-buttton" id='design-title-button' onClick={handleRename}>
                                {!isNameEditing ? <FaRegPenToSquare /> : <FaCheck />} 
                            </button>
                        </div>
                        <div className="toolbar-buttons">
                            <button className="toolbar-button" onClick={handleSaveImage}>
                                <FaDownload />
                            </button>
                            <button className="toolbar-button" onClick={handleUndo} disabled={currentDesign.currentImageIndex == 0}>
                                <FaArrowRotateLeft />
                            </button>
                            <button className="toolbar-button" onClick={() => {setIsNextVisible(true)}} disabled={forwardDisabled()}>
                                <FaArrowRotateRight />
                            </button>
                        </div>
                    </div>
                    <div className="current-image-area">
                        <img className='presented-image-design' src={currentDesign && currentDesign.images && currentDesign.images[currentDesign.currentImageIndex].imageUrl}/>
                    </div>
                    <div className="prompt-area">
                        <input 
                            className="prompt-input" 
                            type="text" 
                            value={promptText} 
                            placeholder="Enter your prompt..."
                            onChange={(e) => setPromptText(e.target.value)} />
                        <button className="button-send" onClick={handlePromptSend} disabled={promptText == ''}>
                            <FaPaperPlane />
                        </button>
                    </div>
                </div>
            }
            {isNextVisible && 
                <div className="current-design-area forward-design-area">
                    <div className="toolbar-forward-design-area">
                        <button id="close-forward-design-area-button" onClick={() => {setIsNextVisible(false)}}>
                            <FaX />
                        </button>
                    </div>
                    <div className="forward-area">
                        {currentDesign && currentDesign.images && currentDesign.images[currentDesign.currentImageIndex].children?.map(image => (
                            <ForwardDesignCard 
                                key={image.imageUrl} 
                                imageUrl={image.imageUrl} 
                                prompt={image.creationPrompt || ""} 
                                onClick={() => handleForwardClose(image.imageUrl)} 
                            />
                        ))}
                    </div>
                </div>
            }
        </div>
    )
}
