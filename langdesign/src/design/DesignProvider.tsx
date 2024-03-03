import React, { useCallback, useContext, useEffect, useState } from 'react';
import axios from 'axios';
import PropTypes from 'prop-types';
import { getLogger } from '../core';
import { storage } from '../configs/firebase';
import { ref, uploadBytes, getDownloadURL } from 'firebase/storage'
import { set, ref as dbRef, push, get } from 'firebase/database'
import { v4 } from 'uuid';
import { AuthContext } from '../auth/AuthProvider';
import { database } from '../configs/firebase';

const log = getLogger('DesignProvider');

type UploadFn = (image: File, uid: string) => Promise<string>;
type DesignFn = (image: string, prompt: string, uid: string) => Promise<string>;
type UndoFn = () => void;
type ForwardFn = (imageUrl: string) => void;
type LoadDesignFn = (designCode: string, designName: string) => void;
type StartNewFn = () => void;
type RenameFn = (name: string, designCode: string) => void;

export interface DesignImage {
    imageUrl: string;
    parentImageUrl: string;
    creationPrompt?: string;
    children?: DesignImage[];
}

export interface DesignProps {
    name: string,
    code: string,
    lastUpdated: number 
    images?: DesignImage[]
    currentImageIndex: number;
}

export interface HistoryDesignProps {
    code: string;
    name: string;
}

export interface DesignsState {
    designs: HistoryDesignProps[];
    currentDesign: DesignProps;
    upload?: UploadFn;
    design?: DesignFn;
    undo?: UndoFn;
    forward?: ForwardFn;
    loadDesign?: LoadDesignFn;
    startNew?: StartNewFn;
    rename?: RenameFn;
}

const initialState: DesignsState = {
    designs: [],
    currentDesign: {name: '', code: '', lastUpdated: 1, currentImageIndex: 0}
};

export const DesignContext = React.createContext<DesignsState>(initialState);

interface DesignProviderProps {
    children: PropTypes.ReactNodeLike,
}

export const DesignProvider: React.FC<DesignProviderProps> = ({ children }) => {
    const [state, setState] = useState<DesignsState>(initialState);
    const {uid, isAuthenticated} = useContext(AuthContext);
    const { currentDesign, designs } = state;
    const upload = useCallback<UploadFn>(uploadCallback, [designs, uid]);
    const design = useCallback<DesignFn>(designCallback, [currentDesign]);
    const undo = useCallback<UndoFn>(undoCallback, [currentDesign]);
    const forward = useCallback<ForwardFn>(forwardCallback, [currentDesign]);
    const loadDesign = useCallback<LoadDesignFn>(loadDesignCallback, [currentDesign]);
    const startNew = useCallback<StartNewFn>(startNewCallback, [currentDesign]);
    const rename = useCallback<RenameFn>(renameCallback, [currentDesign]);
    const value = { currentDesign, designs, upload, design, undo, loadDesign, startNew, forward, rename };
    const newDesignName = 'New design';
    log('render');

    useEffect(() => {
        if(!isAuthenticated) {
            setState(initialState)
        } else {
            fetchUserDesigns();
        }
    }, [isAuthenticated])

    return (
        <DesignContext.Provider value={value}>
            {children}
        </DesignContext.Provider>
    );

    async function fetchUserDesigns() {
        try {
            const snapshot = await get(dbRef(database, `users/${uid}/designs`));
            if (snapshot.exists()) {
                const designsData = snapshot.val();
                const designsArray = Object.keys(designsData).map(key => ({
                    code: designsData[key].code,
                    name: designsData[key].name
                }));
                setState(prevState => ({
                    ...prevState,
                    designs: designsArray.reverse()
                }));
            } else {
                console.log("No designs found for the user", uid);
            }
        } catch (error) {
            console.error("Error fetching user designs:", error);
        }
    }

    function getCurrentTimestamp(): number {
        return Math.floor(Date.now() / 1000)
    }

    function createDesign(firstImageUrl: string) {
        const designCode = uid + '_' + v4();
        const lastUpdated = getCurrentTimestamp()

        set(dbRef(database, 'designs/' + designCode), {
            images: [{imageUrl: firstImageUrl, parentImageUrl: 'startImage'}],
            name: newDesignName,
            lastUpdated: lastUpdated
        });

        setState(prevState => ({
            ...prevState,
            designs: [
                { code:designCode, name: newDesignName, lastUpdated: lastUpdated, currentImageIndex: 0 },
                ...prevState.designs,
            ],
            currentDesign: { 
                name: newDesignName, 
                code: designCode,
                lastUpdated: lastUpdated, 
                currentImageIndex: 0, 
                images: [{ imageUrl: firstImageUrl, parentImageUrl: 'startImage' }], 
                children: [] 
            }
        }));

        return designCode
    }

    function addDesignToUser(designCode: string) {
        const userDesignsRef = dbRef(database, `users/${uid}/designs`);
        const newDesign = push(userDesignsRef);
        set(newDesign, {
            code: designCode,
            name: newDesignName,
        })
    }

    function startNewCallback() {
        setState(prevState => ({
            ...prevState,
            currentDesign: initialState.currentDesign
        }));
    }

    function uploadCallback(image: File, uid: string): Promise<string> {
        const imageRef = ref(storage, `designs/${uid}/${image.name + v4()}`);
        return new Promise((resolve, reject) => {
            uploadBytes(imageRef, image).then((snapshot) => {
                getDownloadURL(snapshot.ref).then((url) => {
                    resolve(url); 
                    const designCode = createDesign(url)
                    addDesignToUser(designCode)
                }).catch(reject); 
            }).catch(reject); 
        });
    }

    function forwardCallback(imageUrl: string): void {
        const forwardIndex = currentDesign.images?.findIndex(image => currentDesign.images && image.imageUrl == imageUrl) || 0
        setState(prevState => ({
            ...prevState,
            currentDesign: { 
                ...prevState.currentDesign,
                currentImageIndex: forwardIndex, 
            }
        }));
    }

    async function renameCallback(name: string, designCode: string): Promise<void> {
        const userDesignsRef = dbRef(database, `users/${uid}/designs`);
        try {
            const userDesignsSnapshot = await get(userDesignsRef);
            const userDesings = userDesignsSnapshot.val();
    
            const designIndex = Object.keys(userDesings).findIndex(key => userDesings[key].code === designCode);
    
            if (designIndex !== -1) {
                const designKey = Object.keys(userDesings)[designIndex];
                userDesings[designKey].name = name;
    
                await set(userDesignsRef, userDesings);
    
                setState(prevState => ({
                    ...prevState,
                    designs: prevState.designs.map(design => {
                        if (design.code === designCode) {
                            return { ...design, name: name };
                        }
                        return design;
                    }),
                    currentDesign: { ...prevState.currentDesign, name: name }
                }));
            } else {
                console.error('Design not found.');
            }
        } catch (error) {
            console.error('Error renaming design:', error);
            throw error;
        }
    };
    

    async function loadDesignCallback(designCode: string, designName: string) {
        try {
            const snapshot = await get(dbRef(database, `designs/${designCode}`));
            if (snapshot.exists()) {
                const designData = snapshot.val();
                setState(prevState => ({
                    ...prevState,
                    currentDesign: { 
                        ...prevState.currentDesign, 
                        currentImageIndex: designData.images.length - 1,
                        lastUpdated: designData.lastUpdated,
                        images: designData.images,
                        name: designName,
                        code: designCode,
                    }
                }));
            }
        } catch (error) {
            console.error("Error fetching user designs:", error);
        }
    }

    function undoCallback(): void {
        const undoIndex = currentDesign.images?.findIndex(image => currentDesign.images && image.imageUrl == currentDesign.images[currentDesign.currentImageIndex].parentImageUrl) || 0
        console.log('undo', undoIndex);
        setState(prevState => ({
            ...prevState,
            currentDesign: { 
                ...prevState.currentDesign, 
                currentImageIndex: undoIndex,
            }
        }));
    }
    
    async function getImageFileFromUrl(imageUrl: string): Promise<File> {
        try {
            const response = await fetch(imageUrl);
            const blob = await response.blob();
    
            const fileName = imageUrl.substring(imageUrl.lastIndexOf('/') + 1);
    
            const file = new File([blob], fileName, { type: blob.type });
            return file;
        } catch (error) {
            console.error('Error fetching image:', error);
            throw error;
        }
    }

    async function addImageToDesign(designCode: string, imageUrl: string, parentImageUrl: string, prompt: string) {
        const designRef = dbRef(database, 'designs/' + designCode);
        
        try {
            //const designSnapshot = await get(designRef);
            //const currentDesign = designSnapshot.val();
            const lastUpdated = getCurrentTimestamp()
            
            if (currentDesign !== null) {
                const currentImages: DesignImage[] = currentDesign.images || [];
                const parentIndex = currentDesign.currentImageIndex;
                
                const newImage = { imageUrl: imageUrl, parentImageUrl: parentImageUrl, children: [], creationPrompt: prompt}
                const updatedImages = [
                    ...currentImages,
                    newImage
                ];
                
                // add the result image as children to the current image
                const children = updatedImages[parentIndex].children
                if(children) {
                    const newChildren = [...children, newImage]
                    updatedImages[parentIndex] = {
                        ...updatedImages[parentIndex], 
                        children: newChildren
                    }
                } else {
                    updatedImages[parentIndex] = {
                        ...updatedImages[parentIndex], 
                        children: [newImage] 
                    }
                }

                await set(designRef, {
                    images: updatedImages,
                    name: currentDesign.name,
                    lastUpdated: lastUpdated,
                });
                setState(prevState => ({
                    ...prevState,
                    currentDesign: { code: prevState.currentDesign.code, name: prevState.currentDesign.name, lastUpdated: lastUpdated, currentImageIndex: updatedImages.length - 1, images: updatedImages }
                }));
                
            } else {
                console.error('Design document does not exist');
            }
        } catch (error) {
            console.error('Error adding image to design:', error);
            throw error;
        }
    }
    
    async function sendForDesign(image: File, prompt: string): Promise<any> {
        const formData = new FormData();
        formData.append('image', image);
    
        try {
            const response = await axios.post('http://127.0.0.1:5000/design', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                },
                params: {
                    prompt: prompt
                },
                responseType: 'arraybuffer'
            });
            const blob = new Blob([response.data], { type: 'image/jpeg' });
            console.log('Request successful')
    
            return blob;
        } catch (error) {
            console.error('Error:', error);
            throw error; 
        }
    }
    
    function designCallback(imageUrl: string, prompt: string, uid: string): Promise<string> {
        console.log('called design callback')
        return new Promise(async (resolve, reject) => {
            try {
                const imageFile = await getImageFileFromUrl(imageUrl);
                const designedImageBlob = await sendForDesign(imageFile, prompt);
                
                const imageRef = ref(storage, `designs/${uid}/${imageFile.name + v4()}`);
                uploadBytes(imageRef, designedImageBlob).then((snapshot) => {
                    getDownloadURL(snapshot.ref).then((url) => {
                        addImageToDesign(currentDesign?.code || "", url, imageUrl, prompt);
                        resolve(url); 
                    }).catch(reject); 
                }).catch(reject);
            } catch (error) {
                reject(error);
            }
        });
    }
};